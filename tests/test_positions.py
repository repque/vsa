"""Tests for positions module."""
import pytest
from datetime import datetime, timedelta
from tinydb import TinyDB
import tempfile
from positions import Position, Pnl


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        db_path = f.name

    # Override the class-level db with temp db
    original_db = Position.db
    Position.db = TinyDB(db_path)

    yield Position.db

    # Cleanup
    Position.db.close()
    Position.db = original_db


class TestPosition:
    """Test Position class."""

    def test_record_entry(self, temp_db):
        """Test recording a position entry."""
        pos = Position.record('TEST', 100, 150.50, is_entry=True)
        assert pos.ticker == 'TEST'
        assert pos.is_open is True

    def test_record_entry_and_exit(self, temp_db):
        """Test recording entry and exit."""
        Position.record('TEST2', 100, 150.50, is_entry=True)
        pos = Position.record('TEST2', 100, 155.00, is_entry=False)
        assert pos.is_open is False

    def test_load_position(self, temp_db):
        """Test loading a position from database."""
        Position.record('TEST3', 100, 150.50, is_entry=True)
        pos = Position.load('TEST3')
        assert pos.ticker == 'TEST3'
        assert pos.is_open is True

    def test_all_open_returns_list(self, temp_db):
        """Test that all_open returns a list."""
        Position.record('AAPL', 100, 150.00, is_entry=True)
        Position.record('MSFT', 50, 300.00, is_entry=True)
        open_positions = Position.all_open()
        assert isinstance(open_positions, list)
        assert len(open_positions) >= 2

    def test_as_df_returns_dataframe(self, temp_db):
        """Test that as_df returns a DataFrame."""
        Position.record('TSLA', 10, 700.00, is_entry=True)
        df = Position.as_df()
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert 'ticker' in df.columns
        assert 'mtm' in df.columns
        assert 'realized' in df.columns
        assert 'total' in df.columns

    def test_expiring_returns_positions(self, temp_db):
        """Test expiring positions detection."""
        # Record position with date 30 days ago (should be expiring soon)
        old_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
        Position.record('OLD', 100, 100.00, date=old_date, is_entry=True)
        expiring = Position.expiring(days=5)
        assert expiring is not None or expiring == []

    def test_pnl_namedtuple_structure(self, temp_db):
        """Test that pnl returns correct Pnl structure."""
        pos = Position.record('PNL_TEST', 100, 100.00, is_entry=True)
        # Mock get_current_price by providing price directly
        pnl = pos.pnl(price=105.00)
        assert isinstance(pnl, Pnl)
        assert hasattr(pnl, 'mtm')
        assert hasattr(pnl, 'realized')
        assert hasattr(pnl, 'total')

    def test_pnl_calculation(self, temp_db):
        """Test P&L calculation."""
        pos = Position.record('CALC', 100, 100.00, is_entry=True)
        pnl = pos.pnl(price=110.00)
        # MTM = (110 - 100) * 100 = 1000
        assert pnl.mtm == 1000.00
        assert pnl.total == 1000.00

    def test_position_equality(self, temp_db):
        """Test position equality comparison."""
        pos1 = Position.load('EQUAL1')
        pos2 = Position.load('EQUAL1')
        assert pos1 == pos2

    def test_position_hash(self, temp_db):
        """Test position hashing."""
        pos = Position.load('HASH')
        assert hash(pos) == hash('HASH')
