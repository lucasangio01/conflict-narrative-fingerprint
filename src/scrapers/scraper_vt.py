import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin


class KyivPost:
    def __init__(self):
        self.start_url = "https://vtforeignpolicy.com/category/wars/ukraine-war/?filter_by=popular"
        self.base_url = "https://www.kp.ru/daily/"
        self.semaphore = asyncio.Semaphore(10)


    async def fetch(self, session, url):
        async with self.semaphore:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()


    async def extract_data(self, session):
        all_urls = []
        url = self.start_url
        html = await self.fetch(session, url)
        soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
        urls = [urljoin(self.base_url, a.find("a")["href"]) for a in soup.find_all("div", attrs = {"class": "td-module-thumb"})]
        all_urls.extend(urls)
        self.df_titles = pd.DataFrame({"url": all_urls})


    async def extract_text(self, session):
        tasks = [self.fetch(session, row.url) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages]
        soups = await asyncio.gather(*parse_tasks)
        all_texts = []
        all_titles = []
        for soup in soups:
            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            title = soup.find("h1", attrs = {"class": "entry-title"}).text.strip()
            all_texts.append(" ".join(paragraphs))
            all_titles.append(title)

        self.df_text = pd.DataFrame({"title": all_titles, "text": all_texts})
        self.df_text.to_csv("../../data/russia/vt.csv")
    
    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)



async def main():
    scraper = KyivPost()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
