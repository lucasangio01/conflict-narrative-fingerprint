import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin



class JPost:
    def __init__(self):
        self.base_url = "https://www.jpost.com/opinion"
        self.semaphore = asyncio.Semaphore(3)


    async def fetch(self, session, url):
        async with self.semaphore:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()


    async def extract_data(self, session):
        all_titles = []
        all_urls = []
        for i in range(1, 4):
            url = f"https://www.jpost.com/opinion?page={i}"
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            urls = [urljoin(self.base_url, t["href"]) for t in soup.find_all("a", attrs = {"class": "article-item-content-link"}) if t is not None]
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
            title = [t.text.strip() for t in soup.find_all("h1", attrs = {"class": "article-main-title"})]
            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            dates = [d.text.strip() for d in soup.find_all("time", attrs = {"class": "datetime"})]
            all_texts.append(" ".join(paragraphs))
            all_dates.append(" ".join(dates))
            all_titles.append(" ".join(title))

        self.df_text = pd.DataFrame({"title": all_titles, "date": all_dates, "text": all_texts})
        self.df_text.to_csv("../../../data/israel/jpost.csv")
    
    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)



async def main():
    scraper = JPost()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
