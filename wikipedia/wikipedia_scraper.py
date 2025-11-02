#!/usr/bin/env python3
"""
Wikipedia Glossary Scraper with Categorization and Summary Generation

This script scrapes the Wikipedia "Glossary of areas of mathematics" page,
categorizes each mathematical area, generates summaries using GPT,
and saves everything as JSON files with complete information.

Source: https://en.wikipedia.org/wiki/Glossary_of_areas_of_mathematics
"""

import os
import re
import json
import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin, urlparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import GPT model functionality (assuming it's available in the project)
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.model import Model
    from wikipedia.prompts import (
        SYSTEM_PROMPT_CAPABILITY_EVALUATION,
        SYSTEM_PROMPT_CATEGORIZATION,
        get_capability_summary_prompt,
        get_capability_categorization_prompt,
    )
    GPT_AVAILABLE = True
except ImportError:
    logger.warning("GPT model not available. Will use fallback summarization.")
    GPT_AVAILABLE = False


def generate_summary_with_gpt(description: str, model: Model, cache_dir: str = None, capability_name: str = None) -> Tuple[str, bool]:
    """
    Generate a concise summary of a capability description using GPT.

    Args:
        description: The full description to summarize
        model: The GPT model to use for summarization
        cache_dir: Directory to cache summaries (optional)
        capability_name: Name of the capability for caching (optional)

    Returns:
        A tuple of (summary, was_cached)
    """
    # Try to load cached summary first
    if cache_dir and capability_name:
        os.makedirs(cache_dir, exist_ok=True)
        # Sanitize filename by replacing invalid characters
        safe_name = "".join(c for c in capability_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        cache_file = os.path.join(cache_dir, f"summary_{safe_name}.txt")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_summary = f.read().strip()
                logger.debug(f"Loaded cached summary for '{capability_name}'")
                return cached_summary, True
            except Exception as e:
                logger.warning(f"Failed to load cached summary for '{capability_name}': {e}")

    sys_prompt = SYSTEM_PROMPT_CAPABILITY_EVALUATION
    user_prompt = get_capability_summary_prompt(description)

    generation_config = {
        "temperature": 0.3,
        "max_tokens": 200,
        "seed": 42
    }

    try:
        summary, metadata = model.generate(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            generation_config=generation_config
        )
        summary = summary.strip()
        logger.debug(f"Generated summary for '{description[:50]}...' with {metadata['output_tokens']} tokens")

        # Cache the summary if cache_dir is provided
        if cache_dir and capability_name:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                logger.debug(f"Cached summary for '{capability_name}' to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache summary for '{capability_name}': {e}")

        return summary, False
    except Exception as e:
        logger.warning(f"Failed to generate summary with GPT: {e}. Using fallback method.")
        # Fallback to first sentence extraction
        for end_char in ['.', '!', '?']:
            if end_char in description:
                fallback_summary = description.split(end_char)[0] + end_char
                # Cache the fallback summary too
                if cache_dir and capability_name:
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            f.write(fallback_summary)
                        logger.debug(f"Cached fallback summary for '{capability_name}' to {cache_file}")
                    except Exception as cache_e:
                        logger.warning(f"Failed to cache fallback summary for '{capability_name}': {cache_e}")
                return fallback_summary, False
        return description, False


def categorize_capability_with_gpt(description: str, model: Model, cache_dir: str = None, capability_name: str = None) -> Tuple[str, bool]:
    """
    Categorize a capability description using GPT into one of the 10 mathematical areas.

    Args:
        description: The capability description to categorize
        model: The GPT model to use for categorization
        cache_dir: Directory to cache categorizations (optional)
        capability_name: Name of the capability for caching (optional)

    Returns:
        A tuple of (category, was_cached)
    """
    # Try to load cached categorization first
    if cache_dir and capability_name:
        os.makedirs(cache_dir, exist_ok=True)
        safe_name = "".join(c for c in capability_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        cache_file = os.path.join(cache_dir, f"category_{safe_name}.txt")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_category = f.read().strip()
                logger.debug(f"Loaded cached category for '{capability_name}': {cached_category}")
                return cached_category, True
            except Exception as e:
                logger.warning(f"Failed to load cached category for '{capability_name}': {e}")

    sys_prompt = SYSTEM_PROMPT_CATEGORIZATION
    user_prompt = get_capability_categorization_prompt(description)

    generation_config = {
        "temperature": 0.1,
        "max_tokens": 50,
        "seed": 42
    }

    try:
        category, metadata = model.generate(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            generation_config=generation_config
        )
        category = category.strip()
        logger.debug(f"Generated category for '{description[:50]}...': {category}")

        # Cache the category if cache_dir is provided
        if cache_dir and capability_name:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(category)
                logger.debug(f"Cached category for '{capability_name}' to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache category for '{capability_name}': {e}")

        return category, False
    except Exception as e:
        logger.warning(f"Failed to generate category with GPT: {e}. Using fallback category.")
        # Fallback to default category
        fallback_category = "Algebra and Functions"
        if cache_dir and capability_name:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(fallback_category)
                logger.debug(f"Cached fallback category for '{capability_name}' to {cache_file}")
            except Exception as cache_e:
                logger.warning(f"Failed to cache fallback category for '{capability_name}': {cache_e}")
        return fallback_category, False


class WikipediaGlossaryScraper:
    """Scraper for Wikipedia glossary of areas of mathematics with categorization and summarization."""

    def __init__(self, base_url: str, output_dir: str, gpt_model: Model = None):
        """
        Initialize the scraper.

        Args:
            base_url: The Wikipedia glossary URL
            output_dir: Directory to save the scraped files
            gpt_model: Optional GPT model for summarization and categorization
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.gpt_model = gpt_model
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Create cache directories
        self.summary_cache_dir = os.path.join(output_dir, "summary_cache")
        self.category_cache_dir = os.path.join(output_dir, "category_cache")
        os.makedirs(self.summary_cache_dir, exist_ok=True)
        os.makedirs(self.category_cache_dir, exist_ok=True)

    def get_page_content(self) -> BeautifulSoup:
        """
        Fetch and parse the Wikipedia glossary page.

        Returns:
            BeautifulSoup object of the page content
        """
        try:
            logger.info(f"Fetching page: {self.base_url}")
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            logger.info("Successfully fetched and parsed the page")
            return soup

        except requests.RequestException as e:
            logger.error(f"Error fetching page: {e}")
            raise

    def get_page_first_section(self, page_url: str) -> str:
        """
        Visit an individual Wikipedia page and extract the first section (introduction).

        Args:
            page_url: URL of the individual Wikipedia page

        Returns:
            First section text content
        """
        try:
            logger.debug(f"Fetching individual page: {page_url}")
            response = self.session.get(page_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the main content area
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            if not content_div:
                logger.warning(f"Could not find main content for {page_url}")
                return ""

            # Collect all consecutive elements before the first h2 as the intro section
            intro_texts = []
            for child in content_div.children:
                # Only consider tag elements
                if not hasattr(child, 'name') or child.name is None:
                    continue
                # Stop at the first h2 (start of second section)
                if child.name == 'h2':
                    break
                # Capture paragraphs and short intro divs (infobox/sidebar divs are skipped)
                if child.name == 'p':
                    text = child.get_text(' ', strip=True)
                    if text:
                        intro_texts.append(text)
                elif child.name in ('div',):
                    # Some pages wrap first paragraphs in a div; extract contained paragraph texts
                    inner_paras = child.find_all('p', recursive=False)
                    for p in inner_paras:
                        text = p.get_text(' ', strip=True)
                        if text:
                            intro_texts.append(text)

            if intro_texts:
                description = ' '.join(intro_texts)
                # Normalize whitespace
                description = ' '.join(description.split())
                logger.debug(f"Extracted first section from {page_url}: {description[:100]}...")
                return description
            else:
                logger.warning(f"No first section content found for {page_url}")
                return ""

        except Exception as e:
            logger.warning(f"Error fetching individual page {page_url}: {e}")
            return ""

    def extract_glossary_entries(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Extract glossary entries by finding links in definition lists.

        Args:
            soup: BeautifulSoup object of the page content

        Returns:
            List of dictionaries containing name and description for each entry
        """
        entries = []

        # Find the main content area
        content_div = soup.find('div', {'class': 'mw-parser-output'})
        if not content_div:
            logger.error("Could not find main content div")
            return entries

        # Find all definition lists (dl elements)
        dl_elements = content_div.find_all('dl')
        logger.info(f"Found {len(dl_elements)} definition lists")

        # Process each definition list
        for dl in dl_elements:
            # Find all definition terms (dt elements) in this list
            dt_elements = dl.find_all('dt')
            logger.info(f"Found {len(dt_elements)} definition terms in this list")

            # Process each definition term
            for dt in dt_elements:
                # Get the main link (first link) in this definition term
                # This should be the primary mathematical topic
                main_link = dt.find('a', href=True)

                if main_link:
                    href = main_link.get('href', '')
                    text = main_link.get_text(strip=True)

                    # Skip if it's not a Wikipedia article link or if it's too short
                    if (href.startswith('/wiki/') and
                        not href.startswith('/wiki/File:') and
                        not href.startswith('/wiki/Template:') and
                        not href.startswith('/wiki/Category:') and
                        not href.startswith('/wiki/Help:') and
                        not href.startswith('/wiki/Special:') and
                        not href.startswith('/wiki/User:') and
                        not href.startswith('/wiki/Talk:') and
                        not href.startswith('/wiki/User_talk:') and
                        not href.startswith('/wiki/Wikipedia:') and
                        len(text) > 3 and
                        len(text) < 100):  # Reasonable length for topic names

                        try:
                            logger.info(f"Processing: {text}")

                            # Visit the individual page and get the first section
                            page_url = urljoin(self.base_url, href)
                            description = self.get_page_first_section(page_url)

                            if description and len(description) > 50:  # Ensure we have substantial content
                                entries.append({
                                    'name': text,
                                    'description': description,
                                    'page_url': page_url
                                })
                                logger.info(f"+ Successfully extracted description for '{text}'")
                            else:
                                logger.warning(f"- No substantial description found for '{text}'")

                            # Add a small delay to be respectful to Wikipedia
                            time.sleep(0.5)

                        except Exception as e:
                            logger.warning(f"Error processing '{text}': {e}")
                            continue

        logger.info(f"Successfully extracted {len(entries)} mathematical topic descriptions")
        return entries

    def clean_filename(self, name: str) -> str:
        """
        Clean a term name to create a valid filename.

        Args:
            name: The term name to clean

        Returns:
            Cleaned filename
        """
        # Replace special characters and spaces with underscores
        filename = re.sub(r'[^\w\s-]', '', name)
        filename = re.sub(r'[\s_-]+', '_', filename)
        filename = filename.strip('_')

        # Limit length
        if len(filename) > 100:
            filename = filename[:100]

        return filename

    def save_entry_to_file(self, entry: Dict[str, str]) -> bool:
        """
        Save a glossary entry to a JSON file with complete information.

        Args:
            entry: Dictionary containing name, description, summary, and area

        Returns:
            True if successful, False otherwise
        """
        try:
            filename = self.clean_filename(entry['name'])
            if not filename:
                logger.warning(f"Could not create filename for: {entry['name']}")
                return False

            filepath = os.path.join(self.output_dir, f"{filename}.json")
            logger.info(f"Creating file: {filepath}")

            # Create the complete JSON structure
            json_data = {
                "capability_name": entry['name'],
                "description": entry['description'],
                "summary": entry.get('summary', ''),
                "area": entry.get('area', 'Unknown'),
                "source": "Wikipedia Glossary of Areas of Mathematics",
                "url": entry.get('page_url', self.base_url),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved: {filename}.json")
            return True

        except Exception as e:
            logger.error(f"Error saving entry {entry['name']}: {e}")
            return False

    def scrape_and_save(self) -> int:
        """
        Main method to scrape the glossary, categorize, summarize, and save all entries.

        Returns:
            Number of entries successfully saved
        """
        try:
            # Fetch and parse the page
            soup = self.get_page_content()

            # Extract entries
            entries = self.extract_glossary_entries(soup)

            if not entries:
                logger.error("No entries found to save")
                return 0

            logger.info(f"Processing {len(entries)} entries with categorization and summarization...")

            # Process each entry with categorization and summarization
            saved_count = 0
            summary_stats = {"generated": 0, "cached": 0}
            category_stats = {"generated": 0, "cached": 0}

            for i, entry in enumerate(entries):
                logger.info(f"Processing entry {i+1}/{len(entries)}: {entry['name']}")

                # Generate summary if GPT model is available
                if self.gpt_model:
                    summary, summary_was_cached = generate_summary_with_gpt(
                        entry['description'],
                        self.gpt_model,
                        self.summary_cache_dir,
                        entry['name']
                    )
                    entry['summary'] = summary
                    if summary_was_cached:
                        summary_stats["cached"] += 1
                    else:
                        summary_stats["generated"] += 1
                else:
                    # Fallback to first sentence
                    description = entry['description'].strip()
                    summary = description
                    for end_char in ['.', '!', '?']:
                        if end_char in description:
                            summary = description.split(end_char)[0] + end_char
                            break
                    entry['summary'] = summary

                # Categorize if GPT model is available
                if self.gpt_model:
                    category, category_was_cached = categorize_capability_with_gpt(
                        entry['description'],
                        self.gpt_model,
                        self.category_cache_dir,
                        entry['name']
                    )
                    entry['area'] = category
                    if category_was_cached:
                        category_stats["cached"] += 1
                    else:
                        category_stats["generated"] += 1
                else:
                    # Fallback to default category
                    entry['area'] = "Algebra and Functions"

                # Save the complete entry
                logger.info(f"Attempting to save entry: {entry['name']}")
                if self.save_entry_to_file(entry):
                    saved_count += 1
                    logger.info(f"[OK] Successfully saved {entry['name']}")
                else:
                    logger.error(f"[FAIL] Failed to save {entry['name']}")

                # Add a small delay to be respectful to Wikipedia and API limits
                time.sleep(0.2)

                # Log progress every 10 entries
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(entries)} entries processed")

            # Log final statistics
            logger.info(f"Successfully saved {saved_count} out of {len(entries)} entries")
            if self.gpt_model:
                logger.info(f"Summary statistics: {summary_stats['generated']} generated, {summary_stats['cached']} loaded from cache")
                logger.info(f"Category statistics: {category_stats['generated']} generated, {category_stats['cached']} loaded from cache")

            return saved_count

        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            return 0


def main():
    """Main function to run the scraper."""

    # Configuration
    WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Glossary_of_areas_of_mathematics"
    # Save pages in the same directory as the script
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "pages")

    logger.info("Starting Wikipedia Glossary Scraper with Categorization and Summarization")
    logger.info(f"Source URL: {WIKIPEDIA_URL}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Initialize GPT model if available
    gpt_model = None
    if GPT_AVAILABLE:
        try:
            # You can configure the model here
            gpt_model = Model(
                model_name="gpt-3.5-turbo",  # or "gpt-4", "o1-mini", etc.
                model_provider="openai"
            )
            logger.info("[OK] GPT model initialized for categorization and summarization")
        except Exception as e:
            logger.warning(f"Failed to initialize GPT model: {e}. Will use fallback methods.")
            gpt_model = None
    else:
        logger.info("GPT model not available. Will use fallback methods for summarization and categorization.")

    # Create scraper instance
    scraper = WikipediaGlossaryScraper(WIKIPEDIA_URL, OUTPUT_DIR, gpt_model)

    # Run the scraper
    saved_count = scraper.scrape_and_save()

    if saved_count > 0:
        logger.info(f"[OK] Scraping completed successfully! Saved {saved_count} JSON entries.")
        logger.info(f"Each entry contains: capability_name, description, summary, area, source, url, timestamp")
    else:
        logger.error("[FAIL] No entries were saved. Please check the logs for errors.")

    return saved_count


if __name__ == "__main__":
    main()