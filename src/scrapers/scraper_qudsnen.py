import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin


class Palestine:
    def __init__(self, website):
        self.website = website
        self.semaphore = asyncio.Semaphore(10)
        if self.website == "qudsnen":
            self.base_url = "https://qudsnen.co/"
            self.urls_tag = "h5"
            self.urls_class = "card-title"
            self.title_tag = "h1"
            self.title_class = "article-title mb-2"
            self.date_tag = "span"
            self.date_class = "date updated"
            self.max_pages = 57

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
        async with self.semaphore:
            async with session.get(url, headers = self.headers) as response:
                response.raise_for_status()
                return await response.text()


    async def extract_data(self, session):
        all_urls = []   
        for i in range(self.max_pages):
            if self.website == "qudsnen":
                self.start_url = f"https://qudsnen.co/all_posts/category/21?page={i}"
            url = self.start_url
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            urls = [urljoin(self.base_url, a.find("a")["href"]) for a in soup.find_all(self.urls_tag, attrs = {"class": self.urls_class})]
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
            print(soup)
            title = soup.find(self.title_tag, attrs = {"class": self.title_class}).text.strip()
            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            if self.website == "qudsnen":
                dates = [d.text.strip() for d in soup.find_all(self.date_tag) if "202" in d.text.strip()]
            else:
                dates = [t.find("span", attrs = {"aria-hidden": "true"}) for t in soup.find_all(self.date_tag, attrs = {"class": self.date_class})]
            all_texts.append(" ".join(paragraphs))
            all_dates.append(dates)
            all_titles.append(title)

        self.df_text = pd.DataFrame({"title": all_titles, "date": all_dates, "text": all_texts})
        self.df_text.to_csv(f"../../data/palestine/{self.website}.csv")
    
    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)



async def main():
    scraper = Palestine(website = "aljazeera")
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
