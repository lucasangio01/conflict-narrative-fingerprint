import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin


class Algemeiner:
    def __init__(self):
        self.base_url = "https://www.aljazeera.com"
        self.semaphore = asyncio.Semaphore(3)
        self.headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9"}
        

    async def fetch(self, session, url):
        async with self.semaphore:
            async with session.get(url, headers = self.headers) as response:
                response.raise_for_status()
                return await response.text()


    async def extract_data(self, session):
        all_titles = []
        all_urls = []
        for i in range(1, 20):
            url = "https://www.aljazeera.com/opinion/"
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            titles = [u.find("span").text for u in soup.find_all("a", attrs = {"class": "u-clickable-card__link article-card__link"})]
            urls = [urljoin(self.base_url, u["href"]) for u in soup.find_all("a", href = True, attrs = {"class": "u-clickable-card__link article-card__link"})]
            all_titles.extend(titles)
            all_urls.extend(urls)
        self.df_titles = pd.DataFrame({"url": all_urls, "title": all_titles})


    async def extract_text(self, session):
        titles = self.df_titles["title"]
        tasks = [self.fetch(session, row.url) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages]
        soups = await asyncio.gather(*parse_tasks)
        all_texts = []
        all_dates = []
        for soup in soups:
            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            dates = [d.text.strip() for d in soup.find_all("div", attrs = {"class": "date"})]
            all_texts.append(" ".join(paragraphs))
            all_dates.append(" ".join(dates))

        self.df_text = pd.DataFrame({"title": titles, "date": all_dates, "text": all_texts})
        self.df_text.to_csv("../../data/palestine/aljazeera.csv")
    
    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)



async def main():
    scraper = Algemeiner()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
