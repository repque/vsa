"""Tests for features module."""
import pytest
import pandas as pd
import numpy as np
from features import get_ratios, get_y, lagged, make_features


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = {
        'Open': np.random.uniform(100, 110, 50),
        'High': np.random.uniform(110, 120, 50),
        'Low': np.random.uniform(90, 100, 50),
        'Close': np.random.uniform(95, 115, 50),
        'Volume': np.random.uniform(1000000, 5000000, 50)
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure High >= Low and prices are consistent
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    return df


class TestGetRatios:
    """Test get_ratios function."""

    def test_get_ratios_returns_dataframe(self, sample_ohlcv_data):
        """Test that get_ratios returns a DataFrame."""
        result = get_ratios(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)

    def test_get_ratios_has_correct_columns(self, sample_ohlcv_data):
        """Test that get_ratios returns expected columns."""
        result = get_ratios(sample_ohlcv_data)
        expected_columns = ['wb', 'uc', 'dc', 'ub', 'db', 'ib', 'ob', 'hv', 'lv']
        assert all(col in result.columns for col in expected_columns)

    def test_get_ratios_drops_na(self, sample_ohlcv_data):
        """Test that get_ratios drops NA values."""
        result = get_ratios(sample_ohlcv_data)
        assert not result.isnull().any().any()

    def test_get_ratios_with_custom_periods(self, sample_ohlcv_data):
        """Test get_ratios with custom period parameter."""
        result = get_ratios(sample_ohlcv_data, periods=20)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestGetY:
    """Test get_y function."""

    def test_get_y_returns_series(self, sample_ohlcv_data):
        """Test that get_y returns a Series."""
        result = get_y(sample_ohlcv_data)
        assert isinstance(result, pd.Series)

    def test_get_y_contains_binary_values(self, sample_ohlcv_data):
        """Test that get_y returns only 0 and 1."""
        result = get_y(sample_ohlcv_data)
        assert set(result.unique()).issubset({0, 1})

    def test_get_y_correct_length(self, sample_ohlcv_data):
        """Test that get_y returns correct length."""
        result = get_y(sample_ohlcv_data)
        assert len(result) == len(sample_ohlcv_data)


class TestLagged:
    """Test lagged function."""

    def test_lagged_adds_columns(self, sample_ohlcv_data):
        """Test that lagged adds new columns."""
        df = sample_ohlcv_data[['Close']].copy()
        result = lagged(df, periods=3)
        assert 'Close_t_1' in result.columns
        assert 'Close_t_2' in result.columns

    def test_lagged_respects_skip(self, sample_ohlcv_data):
        """Test that lagged skips specified columns."""
        df = sample_ohlcv_data[['Close', 'Volume']].copy()
        result = lagged(df, periods=3, skip=['Volume'])
        assert 'Close_t_1' in result.columns
        assert 'Volume_t_1' not in result.columns

    def test_lagged_with_none_skip(self, sample_ohlcv_data):
        """Test lagged with None skip parameter."""
        df = sample_ohlcv_data[['Close']].copy()
        result = lagged(df, periods=2, skip=None)
        assert 'Close_t_1' in result.columns


class TestMakeFeatures:
    """Test make_features function."""

    def test_make_features_without_label(self, sample_ohlcv_data):
        """Test make_features without labels."""
        result = make_features(sample_ohlcv_data, with_label=False)
        assert isinstance(result, pd.DataFrame)
        assert 'y' not in result.columns

    def test_make_features_with_label(self, sample_ohlcv_data):
        """Test make_features with labels."""
        result = make_features(sample_ohlcv_data, with_label=True)
        assert isinstance(result, pd.DataFrame)
        assert 'y' in result.columns

    def test_make_features_no_na_values(self, sample_ohlcv_data):
        """Test that make_features returns no NA values."""
        result = make_features(sample_ohlcv_data)
        assert not result.isnull().any().any()

    def test_make_features_returns_non_empty(self, sample_ohlcv_data):
        """Test that make_features returns non-empty DataFrame."""
        result = make_features(sample_ohlcv_data)
        assert len(result) > 0
