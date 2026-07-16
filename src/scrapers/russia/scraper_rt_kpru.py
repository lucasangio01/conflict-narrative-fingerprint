import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import os
from src.utils.constants import ScraperConfig, PreprocessingConfig
from src.utils.logging_config import get_logger

logger = get_logger("SCRAPING")


class KPRU:
    def __init__(self):
        self.website = "kpru"
        self.base_url = "https://www.kp.ru/"

        self.semaphore = asyncio.Semaphore(5)
        self.headers = {**ScraperConfig.BROWSER_HEADERS_BASE, "Referer": self.base_url, "Origin": self.base_url}

        # CSV path and existing titles
        self.csv_path = PreprocessingConfig.STAGE_ORIGINAL.format(website=self.website)
        if os.path.exists(self.csv_path):
            self.df_text = pd.read_csv(self.csv_path)
            if "Unnamed: 0" in self.df_text.columns:
                self.df_text = self.df_text.drop(columns=["Unnamed: 0"])
            self.existing_titles = set(self.df_text["title"])
        else:
            self.df_text = pd.DataFrame(columns=["title", "date", "text"])
            self.existing_titles = set()

    async def fetch(self, session, url):
        async with self.semaphore:
            async with session.get(url) as response:
                return await response.json()  # JSON API for list of articles

    async def fetch2(self, session, url):
        async with self.semaphore:
            async with session.get(url, headers=self.headers) as response:
                return await response.text()  # HTML page for full text

    async def extract_data(self):
        all_urls = []
        async with aiohttp.ClientSession(headers=self.headers) as session:
            for i in range(15):
                page_number = 736 - i
                logger.info(f"page num: {page_number}")
                url = f"https://s02.api.yc.kpcdn.net/content/api/1/pages/get.json?pages.direction=page&pages.number={page_number}&pages.spot=0&pages.target.class=111&pages.target.id=0"
                data = await self.fetch(session, url)
                top_childs = data.get("childs", [])

                def get_article_urls(blocks):
                    urls = []
                    for block in blocks:
                        if "@type" in block and "article" in block["@type"] and "issue" in block and "@id" in block:
                            urls.append(f"https://www.kp.ru/daily/{block['issue']}/{block['@id']}/")
                        if "childs" in block:
                            urls.extend(get_article_urls(block["childs"]))
                    return urls

                urls = get_article_urls(top_childs)
                urls = [urljoin(self.base_url, u) for u in urls]
                all_urls.extend(urls)

        self.df_titles = pd.DataFrame({"url": all_urls})

    async def extract_text(self, session):
        tasks = [self.fetch2(session, row.url) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages]
        soups = await asyncio.gather(*parse_tasks)

        all_texts = []
        all_titles = []
        all_dates = []

        for soup in soups:
            title_tag = soup.find("h1", attrs={"class": "sc-j7em19-3 eyeguj"})
            if not title_tag:
                continue
            title = title_tag.text.strip()
            if title in self.existing_titles:
                continue
            date_tag = soup.find("span", attrs = {"class": "sc-j7em19-1 dtkLMY"})
            if not date_tag:
                continue
            date = date_tag.text.strip()

            paragraphs = [p.text.strip() for p in soup.find_all("p")]

            all_titles.append(title)
            all_dates.append(date)
            all_texts.append(" ".join(paragraphs))
            self.existing_titles.add(title)

        if all_titles:
            new_df = pd.DataFrame({"title": all_titles, "date": all_dates, "text": all_texts})
            self.df_text = pd.concat([self.df_text, new_df], ignore_index=True)
            self.df_text.drop_duplicates(subset="title", inplace=True)
            self.df_text.to_csv(self.csv_path, index=False)

    async def process_website(self):
        await self.extract_data()
        async with aiohttp.ClientSession(headers=self.headers) as session:
            await self.extract_text(session)


async def main():
    scraper = KPRU()
    await scraper.process_website()


if __name__ == "__main__":
    asyncio.run(main())
