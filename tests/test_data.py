"""Tests for data module."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from data import make_data_folder, compress_data


class TestMakeDataFolder:
    """Test make_data_folder function."""

    def test_creates_new_folder(self):
        """Test that make_data_folder creates a new folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / 'test_folder'
            make_data_folder(str(test_path))
            assert test_path.exists()
            assert test_path.is_dir()

    def test_creates_nested_folders(self):
        """Test that make_data_folder creates nested folders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / 'parent' / 'child' / 'grandchild'
            make_data_folder(str(test_path))
            assert test_path.exists()
            assert test_path.is_dir()

    def test_does_not_fail_if_exists(self):
        """Test that make_data_folder doesn't fail if folder exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / 'existing'
            test_path.mkdir()
            make_data_folder(str(test_path))  # Should not raise
            assert test_path.exists()


class TestCompressData:
    """Test compress_data function."""

    @pytest.fixture
    def daily_ohlcv_data(self):
        """Create sample daily OHLCV data."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        data = {
            'Open': np.random.uniform(100, 110, 30),
            'High': np.random.uniform(110, 120, 30),
            'Low': np.random.uniform(90, 100, 30),
            'Close': np.random.uniform(95, 115, 30),
            'Volume': np.random.uniform(1000000, 5000000, 30)
        }
        df = pd.DataFrame(data, index=dates)
        # Ensure High >= Low
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        return df

    def test_compress_to_weekly(self, daily_ohlcv_data):
        """Test compression to weekly bars."""
        result = compress_data(daily_ohlcv_data, res='W')
        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(daily_ohlcv_data)
        assert all(col in result.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    def test_compress_to_monthly(self, daily_ohlcv_data):
        """Test compression to monthly bars."""
        result = compress_data(daily_ohlcv_data, res='M')
        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(daily_ohlcv_data)

    def test_compress_preserves_ohlc_logic(self, daily_ohlcv_data):
        """Test that compression preserves OHLC logic."""
        result = compress_data(daily_ohlcv_data, res='W')
        # Each compressed bar should have High >= Low
        assert (result['High'] >= result['Low']).all()

    def test_compress_sums_volume(self, daily_ohlcv_data):
        """Test that compression sums volume correctly."""
        result = compress_data(daily_ohlcv_data, res='M')
        # Monthly volume should be greater than any single day
        assert result['Volume'].iloc[0] >= daily_ohlcv_data['Volume'].max()
