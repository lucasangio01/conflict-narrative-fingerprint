import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import os


class UkPravda:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(2)

        self.start_url = "https://www.pravda.com.ua/eng/columns/"
        self.base_url = "https://www.pravda.com.ua/"
        self.urls_tag = "a"
        self.urls_class = "atom-text normal-text"
        self.title_tag = "h1"
        self.title_class = "post_article_title"
        self.date_tag = "span"
        self.date_class = "post_author_date"
        self.max_pages = 10

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
            "X-Requested-With": "XMLHttpRequest",
        }

        # CSV path and existing titles
        self.csv_path = "../../../data/ukraine/ukpravda_original.csv"
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
            await asyncio.sleep(2)  # small delay to be polite
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.text()

    async def extract_data(self, session):
        all_urls = []
        for i in range(1, self.max_pages):
            url = f"https://www.pravda.com.ua/eng/columns/?page={i}"
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            urls = [
                a["href"].replace("\\", "").replace('"', "")
                for a in soup.find_all("a")
                if "href" in a.attrs and "eurointegration" not in a["href"]
            ]
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
    scraper = UkPravda()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
