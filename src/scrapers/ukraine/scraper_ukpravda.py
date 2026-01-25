import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import re


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


    async def fetch(self, session, url):
        async with self.semaphore:
            await asyncio.sleep(5)
            async with session.get(url, headers = self.headers) as response:
                response.raise_for_status()
                return await response.text()


    async def extract_data(self, session):
        all_urls = []   
        for i in range(1, self.max_pages):
            url = f"https://www.pravda.com.ua/eng/columns/?page={i}"
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            urls = [a["href"].replace("\\", "").replace('"', '') for a in soup.find_all("a") if "eurointegration" not in a["href"]]
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
            title = soup.find(self.title_tag, attrs = {"class": self.title_class}).text.strip()
            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            dates = [soup.find(self.date_tag, attrs = {"class": self.date_class}).text.strip()]
            all_texts.append(" ".join(paragraphs))
            all_dates.append(dates)
            all_titles.append(title)

        self.df_text = pd.DataFrame({"title": all_titles, "date": all_dates, "text": all_texts})
        self.df_text.to_csv(f"../../../data/ukraine/ukpravda.csv")
    
    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)



async def main():
    scraper = UkPravda()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
