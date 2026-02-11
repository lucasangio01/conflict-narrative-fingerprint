import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import os


class Russia:
    def __init__(self, website):
        self.website = website
        self.semaphore = asyncio.Semaphore(10)

        if self.website == "vt":
            self.start_url = "https://vtforeignpolicy.com/category/wars/ukraine-war/?filter_by=popular"
            self.base_url = "https://vtforeignpolicy.com"
            self.urls_tag = "div"
            self.urls_class = "td-module-thumb"
            self.title_tag = "h1"
            self.title_class = "entry-title"
            self.date_tag = "time"
            self.date_class = "entry-date updated td-module-date"
            self.max_pages = 2
        elif self.website == "kpru":
            self.start_url = "https://www.kp.ru/daily/allcolumn/"
            self.base_url = "https://www.kp.ru/daily/"
            self.urls_tag = "div"
            self.urls_class = "sc-1tputnk-12"
            self.title_tag = "h1"
            self.title_class = "sc-j7em19-3 eyeguj"
            self.date_tag = ""
            self.date_class = ""
            self.max_pages = 2
        elif self.website == "ria":
            self.start_url = "https://crimea.ria.ru/opinions/"
            self.base_url = "https://crimea.ria.ru/"
            self.urls_tag = "div"
            self.urls_class = "list-item__content"
            self.title_tag = "div"
            self.title_class = "article__title"
            self.date_tag = "div"
            self.date_class = "article__info-date"
            self.max_pages = 2

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Connection": "keep-alive",
            "Referer": self.base_url,
            "Origin": self.base_url,
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
        }

        # CSV path and existing titles
        self.csv_path = f"../../../data/russia/{self.website}_original.csv"
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
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.text()

    async def extract_data(self, session):
        all_urls = []
        for i in range(self.max_pages):
            url = self.start_url
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            urls = [urljoin(self.base_url, a.find("a")["href"]) for a in soup.find_all(self.urls_tag, attrs={"class": self.urls_class})]
            all_urls.extend(urls)
        self.df_titles = pd.DataFrame({"url": all_urls})

    async def extract_text(self, session):
        tasks = [self.fetch(session, row.url) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages]
        soups = await asyncio.gather(*parse_tasks)

        all_texts = []
        all_dates = []
        all_titles = []

        for soup in soups:
            title_tag = soup.find(self.title_tag, attrs={"class": self.title_class})
            if not title_tag:
                continue
            title = title_tag.text.strip()
            if title in self.existing_titles:
                continue

            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            date_tag = soup.find(self.date_tag, attrs={"class": self.date_class}) if self.date_tag else None
            date = date_tag.text.strip() if date_tag else "N/A"

            all_titles.append(title)
            all_texts.append(" ".join(paragraphs))
            all_dates.append(date)
            self.existing_titles.add(title)

        if all_titles:
            new_df = pd.DataFrame({"title": all_titles, "date": all_dates, "text": all_texts})
            self.df_text = pd.concat([self.df_text, new_df], ignore_index=True)
            self.df_text.drop_duplicates(subset="title", inplace=True)
            self.df_text.to_csv(self.csv_path, index=False)

    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)


async def main():
    scraper = Russia(website="rt")
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
