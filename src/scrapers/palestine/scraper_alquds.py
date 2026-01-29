import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin



class LigaNET:
    def __init__(self):
        self.base_url = "https://www.alquds.com/"
        self.semaphore = asyncio.Semaphore(2)
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


    async def fetch(self, session, url):
        await asyncio.sleep(20)
        async with self.semaphore:
            async with session.get(url, headers = self.headers) as response:
                response.raise_for_status()
                return await response.text()


    async def extract_data(self, session):
        all_urls = []
        for i in range(1, 3):
            url = f"https://www.alquds.com/en/categories/opinions.turbo_stream?page={i}"
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            urls = [urljoin(self.base_url, t["href"]) for t in soup.find_all("a", attrs = {"class": "hover:text-primary transition-colors"}) if t is not None]
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
            title = [t.text.strip() for t in soup.find_all("h1", attrs = {"class": "mb-0 text-2xl font-quds-bold antialiased text-base-content mt-2"})]
            paragraphs = [p.text.strip() for p in soup.find_all("p") if p.get("style") in ["text-align:justify;", "text-align:right;"]]
            if len(paragraphs) == 0:
                paragraphs = [p.text.strip() for p in soup.find_all("div", attrs = {"class": "content-section my-6"}) if len(p.text.strip()) > 10]
            dates = [d.text.strip() for d in soup.find_all("span", attrs = {"class": "ml-1 text-sm"})]
            all_texts.append(" ".join(paragraphs))
            all_dates.append(" ".join(dates))
            all_titles.append(" ".join(title))

        self.df_text = pd.DataFrame({"title": all_titles, "date": all_dates, "text": all_texts})
        self.df_text.to_csv("../../data/palestine/1_original/alquds.csv")
    

    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)



async def main():
    scraper = LigaNET()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
