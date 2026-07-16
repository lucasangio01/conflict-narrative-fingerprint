import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import os
from src.utils.constants import ScraperConfig, PreprocessingConfig


class YnetGlobal:
    def __init__(self):
        self.base_url = "https://www.ynetnews.com/opinions-analysis"
        self.semaphore = asyncio.Semaphore(3)
        self.headers = {**ScraperConfig.BROWSER_HEADERS_BASE, "Referer": self.base_url, "Origin": self.base_url}

        self.csv_path = PreprocessingConfig.STAGE_ORIGINAL.format(website="ynet_global")
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
            await asyncio.sleep(1)
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.text()

    async def extract_data(self, session):
        url = self.base_url
        html = await self.fetch(session, url)
        soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
        urls = [urljoin(self.base_url, g.find("a")["href"]) for g in soup.find_all("div", attrs={"class": "slotTitle"})]
        self.df_titles = pd.DataFrame({"url": urls})

    async def extract_text(self, session):
        tasks = [self.fetch(session, row.url) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages]
        soups = await asyncio.gather(*parse_tasks)

        all_texts = []
        all_dates = []
        all_titles = []

        for soup in soups:
            paragraphs = [p.text.strip() for p in soup.find_all("span", attrs={"data-text": "true"})]
            titles = [a.text.strip() for a in soup.find_all("h1", attrs={"class": "mainTitle"})]
            dates = [time_tag.get("datetime", "").strip() for time_tag in soup.select("span.date time")]
            dates = [dt if dt else "N/A" for dt in dates]
            title = " ".join(titles)
            if title not in self.existing_titles:
                all_titles.append(title)
                all_texts.append(" ".join(paragraphs))
                all_dates.append(" ".join(dates))
                self.existing_titles.add(title)

        if all_titles:
            new_df = pd.DataFrame({"title": all_titles, "date": all_dates, "text": all_texts})
            self.df_text = pd.concat([self.df_text, new_df], ignore_index=True).sort_values(by="date", ascending=False)
            self.df_text.drop_duplicates(subset="title", inplace=True)
            self.df_text.to_csv(self.csv_path, index=False)

    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)


async def main():
    scraper = YnetGlobal()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
