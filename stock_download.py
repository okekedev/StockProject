from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import pandas as pd
import os
from datetime import datetime
import time
import glob

def download_and_get_data():
    output_dir = "./stock_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nasdaq_screener.csv")
    
    # Delete only the existing nasdaq_screener.csv if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file: {output_path}")
    
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')  # Updated headless mode
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--ignore-certificate-errors')
    
    prefs = {
        "download.default_directory": os.path.abspath(output_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        driver = None
        try:
            service = Service()
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
            })
            
            print("Opening NASDAQ website...")
            driver.get("https://www.nasdaq.com/market-activity/stocks/screener")
            
            wait = WebDriverWait(driver, 30)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            print("Looking for download button...")
            button_selectors = [
                "button.jupiter22-c-table__download-csv",
                ".jupiter22-c-table__download-csv",
                "//button[contains(@class, 'jupiter22-c-table__download-csv')]"
            ]
            
            download_button = None
            for selector in button_selectors:
                try:
                    if selector.startswith("//"):
                        download_button = wait.until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        download_button = wait.until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                    if download_button:
                        break
                except:
                    continue
            
            if not download_button:
                raise Exception("Could not find download button")
                
            # List existing CSV files in the directory before the download
            existing_files = set(glob.glob(os.path.join(output_dir, "*.csv")))
            print("Existing CSV files before download:", existing_files)
            
            print("Clicking download button...")
            driver.execute_script("arguments[0].click();", download_button)
            
            print("Waiting for download...")
            timeout = 60
            start_time = datetime.now()
            downloaded_file = None
            
            while (datetime.now() - start_time).seconds < timeout:
                # List all CSV files after the download attempt
                current_files = set(glob.glob(os.path.join(output_dir, "*.csv")))
                # Find new files by comparing with existing files
                new_files = current_files - existing_files
                
                if new_files:
                    downloaded_file = new_files.pop()
                    # Ensure we don't pick up unrelated files like test_stock_accuracy.csv
                    # Check if the filename suggests it's the NASDAQ screener file
                    if ("nasdaq" in downloaded_file.lower() or "screener" in downloaded_file.lower()) and "test_stock_accuracy" not in downloaded_file.lower():
                        print(f"Found downloaded file: {downloaded_file}")
                        # Move the downloaded file to nasdaq_screener.csv
                        os.rename(downloaded_file, output_path)
                        print(f"Moved to: {output_path}")
                        df = pd.read_csv(output_path)
                        # Ensure 'IPO Year' column if 'IPOyear' exists
                        if 'IPOyear' in df.columns:
                            df = df.rename(columns={'IPOyear': 'IPO Year'})
                        # Add Data_Date column
                        df['Data_Date'] = pd.to_datetime('today').strftime('%Y-%m-%d')
                        df.to_csv(output_path, index=False)
                        print(f"Pandas read {len(df)} records")
                        return df
                    else:
                        print(f"Ignoring unrelated file: {downloaded_file}")
                time.sleep(1)
            
            raise Exception("Download timed out or no valid NASDAQ screener file found")
        
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Max retries reached, no data retrieved.")
                return pd.DataFrame()
        
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

print("Starting data download...")
dataset = download_and_get_data()

if dataset.empty:
    print("No data retrieved.")
    dataset = pd.DataFrame(columns=[
        'Symbol', 'Name', 'Last Sale', 'Volume', 
        'Market Cap', 'Sector', 'Industry', 'Data_Date'
    ])
else:
    print(f"Successfully retrieved {len(dataset)} records.")
    print("First few rows:\n", dataset.head())