"""Console demonstration of single-channel versus unified observations."""

from data_gen import generate_dataset
from cross_channel import integrate_channels


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


if __name__ == '__main__':
    main()
