import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import os


class LigaNET:
    def __init__(self):
        self.base_url = "https://www.liga.net/en/opinion"
        self.semaphore = asyncio.Semaphore(3)
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
        self.csv_path = "../../../data/ukraine/liganet_original.csv"
        if os.path.exists(self.csv_path):
            self.df_text = pd.read_csv(self.csv_path)
            if "Unnamed: 0" in self.df_text.columns:
                self.df_text = self.df_text.drop(columns=["Unnamed: 0"])
            self.existing_titles = set(self.df_text["title"])
        else:
            self.df_text = pd.DataFrame(columns=["title", "date", "text"])
            self.existing_titles = set()

    async def fetch(self, session, url):
        await asyncio.sleep(1)
        async with self.semaphore:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.text()

    async def extract_data(self, session):
        all_urls = []
        for i in range(1, 5):
            url = f"https://www.liga.net/en/opinion/page/{i+1}"
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            urls = [
                t["href"] for t in soup.find_all("a", attrs={"class": "idea-card__title is-bold is-accent"})
                if t is not None
            ]
            all_urls.extend(urls)
        self.df_titles = pd.DataFrame({"url": all_urls})

    async def extract_text(self, session):
        tasks = [self.fetch(session, row.url) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages]
        soups = await asyncio.gather(*parse_tasks)

        all_texts = []
        all_titles = []
        all_dates = []

        for soup in soups:
            title_tags = soup.find_all("h1", attrs={"class": "article-header__title is-accent"})
            if not title_tags:
                continue
            title = " ".join([t.text.strip() for t in title_tags])
            if title in self.existing_titles:
                continue

            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            date_tags = soup.find_all("time", attrs={"class": "article-header__date"})
            date = " ".join([d.text.strip() for d in date_tags]) if date_tags else "N/A"

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
    scraper = LigaNET()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
