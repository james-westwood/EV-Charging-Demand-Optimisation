"""Instructions for manual download of ACN-Data."""
from src.logging_config import get_logger

logger = get_logger(__name__)


def print_download_instructions():
    """Print instructions for manual download of ACN-Data."""
    url = "https://ev.caltech.edu/dataset"
    instructions = (
        "\nACN-Data Manual Download Instructions:\n"
        f"1. Visit {url}\n"
        "2. Register for a free account if you haven't already.\n"
        "3. Go to the 'Download' section.\n"
        "4. Select the desired site (e.g., Caltech or JPL) and date range.\n"
        "5. Download the data as a CSV file.\n"
        "6. Place the downloaded CSV in 'energy-forecasting/data/raw/ev_sessions/'.\n"
    )
    print(instructions)


if __name__ == "__main__":
    print_download_instructions()
