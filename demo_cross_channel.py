"""Console demonstration of single-channel versus unified observations."""

from cross_channel import integrate_channels
from data_gen import generate_dataset, to_operational_events
from scoring import score
from signals import compute


def main():
    data = generate_dataset(n_players=240, history_days=182, target_events=12_000, seed=42)
    result = integrate_channels(
        data.players, data.online_casino, data.sports_betting, data.land_based,
        data.start_time, data.end_time,
    )
    comparison = result.unified_summaries.sort_values(
        ['active_channels', 'cross_channel_deposit_uplift'], ascending=False
    ).head(8)
    columns = [
        'activity_entity_id', 'identity_status', 'active_channels',
        'largest_single_channel_deposits', 'total_observed_deposits',
        'cross_channel_deposit_uplift', 'observed_net_losses',
    ]
    print('Synthetic cross-channel comparison (not an OLG schema)')
    print(comparison[columns].to_string(index=False))
    print(f'\nNormalized events: {len(result.events):,}')
    print(f'Duplicate source representations removed: {result.duplicate_records_removed:,}')
    print('Identity statuses:')
    print(result.identity_map.identity_status.value_counts().to_string())

    # Select an illustration using operational cross-channel uplift only.
    # Held scenario labels are not read in this non-validation demonstration.
    operational = to_operational_events(data)
    single = score(compute(data.players, operational, False, False)).set_index('player_id')
    unified = score(compute(data.players, operational, True, True)).set_index('player_id')
    comparison = result.unified_summaries[
        result.unified_summaries.canonical_player_id.notna()
        & result.unified_summaries.active_channels.ge(2)
    ].sort_values('cross_channel_deposit_uplift', ascending=False)
    if len(comparison):
        selected = comparison.iloc[0]
        player_id = selected.canonical_player_id
        channels = result.channel_summaries[
            result.channel_summaries.canonical_player_id.eq(player_id)
        ][['channel', 'observed_deposits', 'observed_net_losses']]
        print('\nOperational example selected by cross-channel uplift:')
        print(f'{player_id} - fictional multi-channel account')
        print(channels.to_string(index=False))
        print(
            f"Unified deposits: ${selected.total_observed_deposits:,.2f}; "
            f"largest channel: ${selected.largest_single_channel_deposits:,.2f}; "
            f"uplift: {selected.cross_channel_deposit_uplift:.2f}x"
        )
        print(
            f"Online-only priority: {single.loc[player_id, 'priority']}; "
            f"unified priority: {unified.loc[player_id, 'priority']}"
        )


if __name__ == '__main__':
    main()
