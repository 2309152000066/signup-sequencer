use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use chrono::Utc;
use sqlx::{Postgres, Transaction};
use tokio::sync::Notify;
use tokio::{select, time};
use tracing::info;

use crate::app::App;
use crate::database::methods::DbMethods;
use crate::database::types::DeletionEntry;
use crate::identity_tree::{Hash, TreeState, TreeVersionReadOps, UnprocessedStatus};
use crate::retry_tx;

// Because tree operations are single threaded (done one by one) we are running
// them from single task that determines which type of operations to run. It is
// done that way to reduce number of used mutexes and eliminate the risk of some
// tasks not being run at all as mutex is not preserving unlock order.
pub async fn modify_tree(app: Arc<App>, wake_up_notify: Arc<Notify>) -> anyhow::Result<()> {
    info!("Starting modify tree task.");

    let batch_deletion_timeout = chrono::Duration::from_std(app.config.app.batch_deletion_timeout)
        .context("Invalid batch deletion timeout duration")?;
    let min_batch_deletion_size = app.config.app.min_batch_deletion_size;

    let mut timer = time::interval(Duration::from_secs(5));

    loop {
        // We wait either for a timer tick or a full batch
        select! {
            _ = timer.tick() => {
                info!("Modify tree task woken due to timeout");
            }

            () = wake_up_notify.notified() => {
                info!("Modify tree task woken due to request");
            },
        }

        let tree_state = app.tree_state()?;

        retry_tx!(&app.database, tx, {
            do_modify_tree(
                &mut tx,
                batch_deletion_timeout,
                min_batch_deletion_size,
                tree_state,
                &wake_up_notify,
            )
            .await
        })
        .await?;

        // wake_up_notify.notify_one();
    }
}

async fn do_modify_tree(
    tx: &mut Transaction<'_, Postgres>,
    batch_deletion_timeout: chrono::Duration,
    min_batch_deletion_size: usize,
    tree_state: &TreeState,
    wake_up_notify: &Arc<Notify>,
) -> anyhow::Result<()> {
    let deletions = get_deletions(
        tx,
        batch_deletion_timeout,
        min_batch_deletion_size,
        tree_state,
    )
    .await?;

    // Deleting identities has precedence over inserting them.
    if !deletions.is_empty() {
        run_deletions(tx, tree_state, deletions).await?;
    } else {
        run_insertions(tx, tree_state).await?;
    }

    // Immediately look for next operations
    wake_up_notify.notify_one();

    Ok(())
}

pub async fn get_deletions(
    tx: &mut Transaction<'_, Postgres>,
    batch_deletion_timeout: chrono::Duration,
    min_batch_deletion_size: usize,
    tree_state: &TreeState,
) -> anyhow::Result<Vec<DeletionEntry>> {
    let deletions = tx.get_deletions().await?;

    if deletions.is_empty() {
        return Ok(Vec::new());
    }

    let last_deletion_timestamp = tx.get_latest_deletion().await?.timestamp;

    // If the minimum deletions batch size is not reached and the deletion time
    // interval has not elapsed then we can skip
    if deletions.len() < min_batch_deletion_size
        && Utc::now() - last_deletion_timestamp <= batch_deletion_timeout
    {
        return Ok(Vec::new());
    }

    // Dedup deletion entries
    let deletions = deletions.into_iter().collect::<HashSet<DeletionEntry>>();
    let mut deletions = deletions.into_iter().collect::<Vec<DeletionEntry>>();

    // Check if the deletion batch could potentially create:
    // - duplicate root on the tree when inserting to identities
    // - duplicate root on batch
    // Such situation may happen only when deletions are done from the last inserted leaf in
    // decreasing order (each next leaf is decreased by 1) - same root for identities, or when
    // deletions are going to create same tree state - continuous deletions.
    // To avoid such situation we sort then in ascending order and only check the scenario when
    // they are continuous ending with last leaf index
    deletions.sort_by(|d1, d2| d1.leaf_index.cmp(&d2.leaf_index));

    if let Some(last_leaf_index) = tree_state.latest_tree().next_leaf().checked_sub(1) {
        let indices_are_continuous = deletions
            .windows(2)
            .all(|w| w[1].leaf_index == w[0].leaf_index + 1);

        if indices_are_continuous && deletions.last().unwrap().leaf_index == last_leaf_index {
            tracing::warn!(
                "Deletion batch could potentially create a duplicate root batch. Deletion \
                 batch will be postponed"
            );
            return Ok(Vec::new());
        }
    }

    Ok(deletions)
}

pub async fn run_insertions(
    tx: &mut Transaction<'_, Postgres>,
    tree_state: &TreeState,
) -> anyhow::Result<()> {
    let unprocessed = tx.get_unprocessed_commitments().await?;
    if unprocessed.is_empty() {
        return Ok(());
    }

    let latest_tree = tree_state.latest_tree();

    let next_leaf = latest_tree.next_leaf();
    let next_db_index = tx.get_next_leaf_index().await?;

    assert_eq!(
        next_leaf, next_db_index,
        "Database and tree are out of sync. Next leaf index in tree is: {next_leaf}, in database: \
         {next_db_index}"
    );

    let mut pre_root = latest_tree.get_root();

    for (idx, identity) in unprocessed.iter().enumerate() {
        let leaf_idx = next_leaf + idx;
        latest_tree.update(leaf_idx, *identity);
        let root = latest_tree.get_root();

        tx.insert_pending_identity(leaf_idx, identity, &root, &pre_root)
            .await
            .expect("Failed to insert identity - tree will be out of sync");

        pre_root = root;
    }

    tx.trim_unprocessed().await?;

    Ok(())
}

pub async fn run_deletions(
    tx: &mut Transaction<'_, Postgres>,
    tree_state: &TreeState,
    mut deletions: Vec<DeletionEntry>,
) -> anyhow::Result<()> {
    let (leaf_indices, previous_commitments): (Vec<usize>, Vec<Hash>) = deletions
        .iter()
        .map(|d| (d.leaf_index, d.commitment))
        .unzip();

    let mut pre_root = tree_state.latest_tree().get_root();
    // Delete the commitments at the target leaf indices in the latest tree,
    // generating the proof for each update
    let data = tree_state.latest_tree().delete_many(&leaf_indices);

    assert_eq!(
        data.len(),
        leaf_indices.len(),
        "Length mismatch when appending identities to tree"
    );

    // Insert the new items into pending identities
    let items = data.into_iter().zip(leaf_indices);
    for ((root, _proof), leaf_index) in items {
        tx.insert_pending_identity(leaf_index, &Hash::ZERO, &root, &pre_root)
            .await?;
        pre_root = root;
    }

    // Remove the previous commitments from the deletions table
    tx.remove_deletions(&previous_commitments).await?;

    Ok(())
}
