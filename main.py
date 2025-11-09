"""Main entry point for VSA predictions and email reporting."""
import argparse
import logging
import sys
from typing import List, Optional

import mail
from model import stocks, make_predictions, train, save
from logger import setup_logging
from exceptions import VSAException
from config import MODEL_PATH

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='VSA Stock Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run predictions and send email
  python main.py

  # Train a new model
  python main.py --train

  # Run predictions without sending email
  python main.py --no-email

  # Use custom stock list
  python main.py --stocks AAPL MSFT TSLA

  # Set log level
  python main.py --log-level DEBUG
        """
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new model instead of running predictions'
    )

    parser.add_argument(
        '--no-email',
        action='store_true',
        help='Generate predictions but do not send email'
    )

    parser.add_argument(
        '--stocks',
        nargs='+',
        help='List of stock tickers to analyze (default: use predefined list)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        help=f'Path to model file (default: {MODEL_PATH})'
    )

    parser.add_argument(
        '--email-to',
        type=str,
        help='Email recipient (default: from config)'
    )

    return parser.parse_args()


def train_model(stock_list: List[str], model_path: str) -> int:
    """Train and save a new model.

    Args:
        stock_list: List of stock tickers
        model_path: Path to save model

    Returns:
        Exit code
    """
    try:
        logger.info(f'[SYSTEM] Training model on {len(stock_list)} stocks')
        model = train(stock_list)

        logger.info(f'[SYSTEM] Saving model to {model_path}')
        save(model, model_path)

        logger.info('[SYSTEM] Model training completed successfully')
        return 0

    except Exception as e:
        logger.exception(f'[SYSTEM] Model training failed: {str(e)}')
        return 1


def run_predictions(stock_list: List[str], send_email: bool, model_path: Optional[str], email_to: Optional[str]) -> int:
    """Run predictions and optionally send email.

    Args:
        stock_list: List of stock tickers
        send_email: Whether to send email report
        model_path: Path to model file
        email_to: Email recipient

    Returns:
        Exit code
    """
    try:
        logger.info(f'[SYSTEM] Generating predictions for {len(stock_list)} stocks')
        pred = sorted(list(make_predictions(stock_list, file_name=model_path)), reverse=True)

        logger.info(f'[SYSTEM] Found {len(pred)} positive predictions')

        if pred:
            for prob, ticker in pred[:10]:  # Show top 10
                logger.info(f'  {ticker}: {prob:.2%}')

        if send_email:
            logger.info('[SYSTEM] Sending email report')
            mail.send(body=pred, to=email_to)
        else:
            logger.info('[SYSTEM] Skipping email (--no-email flag set)')

        logger.info('[SYSTEM] VSA pipeline completed successfully')
        return 0

    except Exception as e:
        logger.exception(f'[SYSTEM] Prediction pipeline failed: {str(e)}')
        return 1


def main() -> int:
    """Run VSA application.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_args()

    # Setup logging with custom level
    import os
    os.environ['LOG_LEVEL'] = args.log_level
    setup_logging()

    try:
        logger.info('[SYSTEM] Starting VSA application')

        # Determine stock list
        stock_list = args.stocks if args.stocks else stocks

        # Determine model path
        model_path = args.model_path or MODEL_PATH

        if args.train:
            return train_model(stock_list, model_path)
        else:
            return run_predictions(stock_list, not args.no_email, model_path, args.email_to)

    except VSAException as e:
        logger.error(f'[SYSTEM] VSA error: {str(e)}')
        return 1
    except Exception as e:
        logger.exception(f'[SYSTEM] Unexpected error: {str(e)}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
