import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd


class KyivPost:
    def __init__(self):
        self.base_url = "https://southfront.press/category/all-articles/products/military-report/ukraine-military-report/"
        self.semaphore = asyncio.Semaphore(1)


    async def fetch(self, session, url):
        async with self.semaphore:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()


    async def extract_data(self, session):
        all_titles = []
        all_urls = []
        for i in range(3): # prendiamo solo le prime 3 pagine
            url = f"https://southfront.press/category/all-articles/products/military-report/ukraine-military-report/page/{i+1}/"
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            titles = [t.find("a").text.strip() for t in soup.find_all("div", attrs = {"class": "title"})]
            urls = [t.find("a")["href"] for t in soup.find_all("div", attrs = {"class": "title"})]
            all_titles.extend(titles)
            all_urls.extend(urls)
            #next_page_button = soup.find("a", attrs = {"aria-label": "Next »", "class": "page-link"})
            # if next_page_button:
            #     next_page_link = next_page_button["href"]
            #     url = next_page_link
            # else:
            #     url = None
        self.df_titles = pd.DataFrame({"url": all_urls, "title": all_titles})


    async def extract_text(self, session):
        titles = self.df_titles["title"]
        tasks = [self.fetch(session, row.url) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages]
        soups = await asyncio.gather(*parse_tasks)
        all_texts = []
        for soup in soups:
            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            all_texts.append(" ".join(paragraphs))

        self.df_text = pd.DataFrame({"title": titles, "text": all_texts})
        self.df_text.to_csv("../../data/russia/southfront.csv")
    
    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)



async def main():
    scraper = KyivPost()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
