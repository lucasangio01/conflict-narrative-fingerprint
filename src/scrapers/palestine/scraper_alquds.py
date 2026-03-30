import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import os
import random

class AlQuds:
    def __init__(self):
        self.base_url = "https://www.alquds.com/"
        self.semaphore = asyncio.Semaphore(2)
        self.csv_path = "../../../data/pal_isr/alquds_original.csv"
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": self.base_url,
        }

        # --- Incremental Logic Setup ---
        if os.path.exists(self.csv_path):
            self.df_text = pd.read_csv(self.csv_path)
            # Cleanup common redundant columns
            for col in ["Unnamed: 0", "id"]:
                if col in self.df_text.columns:
                    self.df_text = self.df_text.drop(columns=[col])
            
            self.existing_titles = set(self.df_text["title"].astype(str))
        else:
            self.df_text = pd.DataFrame(columns=["title", "date", "text"])
            self.existing_titles = set()

    async def fetch(self, session, url, wait=False):
        if wait:
            await asyncio.sleep(random.uniform(15, 25)) 
        
        retries = 3
        for attempt in range(retries):
            async with self.semaphore:
                try:
                    async with session.get(url, headers=self.headers, timeout=30) as response:
                        if response.status == 429:
                            wait_time = 60 * (attempt + 1)
                            print(f"⚠️ 429 Blocked. Sleeping for {wait_time}s on {url}")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        response.raise_for_status()
                        return await response.text()
                
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"❌ Failed to fetch {url} after {retries} attempts: {e}")
                        return None
                    await asyncio.sleep(10)
        return None

    async def extract_data(self, session):
        all_urls = []
        for i in range(1, 3): 
            url = f"https://www.alquds.com/en/categories/opinions.turbo_stream?page={i}"
            html = await self.fetch(session, url, wait=False)
            if not html: continue
            
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            urls = [urljoin(self.base_url, t["href"]) for t in soup.find_all("a", attrs={"class": "hover:text-primary transition-colors"}) if t.has_attr('href')]
            all_urls.extend(urls)
        
        self.df_titles = pd.DataFrame({"url": list(set(all_urls))})

    async def extract_text(self, session):
        # Using gather but keeping the slow fetch internally
        tasks = [self.fetch(session, row.url, wait=True) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages if html]
        soups = await asyncio.gather(*parse_tasks)
        
        new_data = []

        for soup in soups:
            title_tag = soup.find("h1", attrs={"class": "mb-0 text-2xl font-quds-bold antialiased text-base-content mt-2"})
            title = title_tag.text.strip() if title_tag else ""

            if title == "" or title in self.existing_titles:
                continue

            paragraphs = [p.text.strip() for p in soup.find_all("p") if p.get("style") in ["text-align:justify;", "text-align:right;"]]
            if len(paragraphs) == 0:
                paragraphs = [div.text.strip() for div in soup.find_all("div", attrs={"class": "content-section my-6"})]
            
            date_tag = soup.find("span", attrs={"class": "ms-1 text-sm"})
            date = date_tag.text.strip() if date_tag else ""

            new_data.append({"title": title, "date": date, "text": " ".join(paragraphs)})
            self.existing_titles.add(title)

        # --- Handle ID and Saving ---
        if new_data:
            new_df = pd.DataFrame(new_data)
            # Combine and reset index to ensure continuity
            self.df_text = pd.concat([self.df_text, new_df], ignore_index=True)
            
            # Create/Update the 'id' column based on the new total length
            self.df_text['id'] = self.df_text.index
            
            # Move 'id' to the first column
            cols = ['id'] + [c for c in self.df_text.columns if c != 'id']
            self.df_text = self.df_text[cols]
            
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self.df_text.to_csv(self.csv_path, index=False)
            print(f"✅ Added {len(new_data)} new articles. Total rows: {len(self.df_text)}")
        else:
            print("ℹ️ No new articles found.")

    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)

async def main():
    scraper = AlQuds()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)

if __name__ == "__main__":
    asyncio.run(main())