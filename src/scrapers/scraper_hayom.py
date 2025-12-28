import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import re


class IsraelHayom:
    def __init__(self):
        self.base_url = "https://www.israelhayom.com/opinions/"
        self.semaphore = asyncio.Semaphore(3)


    async def fetch(self, session, url):
        async with self.semaphore:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()


    async def extract_data(self, session):
        all_titles = []
        all_urls = []
        for i in range(1, 15):
            url = f"https://www.israelhayom.com/opinions/page/{i}"
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            h3 = [u for u in soup.find_all("h3", attrs = {"class": "jeg_post_title"})]
            titles = [t.find("a").text.strip() for t in h3 if t is not None]
            urls = [t.find("a")["href"] for t in h3 if t is not None]
            all_titles.extend(titles)
            all_urls.extend(urls)
            print(all_titles)
        self.df_titles = pd.DataFrame({"url": all_urls, "title": all_titles})
        self.df_titles.to_csv("../../data/israel/hayom.csv")


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
            dates = [d for d in soup.find_all("span", attrs = {"class": "last_modified_paragraph"})]
            all_texts.append(" ".join(paragraphs))
            all_dates.append(" ".join(dates))

        self.df_text = pd.DataFrame({"title": titles, "date": all_dates, "text": all_texts})
        self.df_text.to_csv("../../data/israel/hayom.csv")
    
    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)



async def main():
    scraper = IsraelHayom()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
