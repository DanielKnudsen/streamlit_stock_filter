import os
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path
from rank import load_config

# Ladda .env-filen endast om den finns
if Path('.env').exists():
    load_dotenv()
    
# Bestäm miljön (default till 'local')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')

def Create_df_tickers(path_file_name:str):

    markets = {'LargeCap':['35207'],
                'MidCap':['35208'],
                'SmallCap':['35209'],
                'FirstNorth':['35186'],
                'FirstNorthPremier':['35187']}
    sectors = {'Bank':'1',
            'Dagligvaror':'2',
            'Energi':'3',
            'Fastigheter':'4',
            'Finansiella tjänster':'5',
            'Försäkring':'6',
            'Hälsovård':'7',
            'Industri':'8',
            'Kemi':'9',
            'Kraft':'10',
            'Råvaror':'11',
            'Sällanköpsvaror':'12',
            'Teknik':'13',
            'Telekommunikation':'14'}
    
    def get_tickers_from_list(links:list):
        prefix = "/bors/aktier/"
        undantag = "investor-relations"
        kort_undantag = "/bors/aktier/" 

        resultat = [item for item in links 
                        if item.startswith(prefix) 
                        and undantag not in item
                        and item != kort_undantag
                        ]
        tickers = [item.split('/')[3].split('-')[0] if len(item.split('/')[3].split('-')) == 1 else '-'.join(item.split('/')[3].split('-')[:-1]) for item in resultat]
        return tickers
        
    def skrapa_uppdelad_tabell(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            tabeller = soup.find_all("table")

            all_data = []

            for tabell in tabeller:
                rader = tabell.find_all("tr")
                for rad in rader:
                    #Hämta endast den första cellen i varje rad
                    första_cell = rad.find(["td", "th"])
                    if första_cell: # Kontrollera att det finns en första cell
                        länk = första_cell.find("a")
                        if länk:
                            länk_text = länk.text.strip()
                            länk_url = länk.get("href")
                            all_data.append({"text": länk_text, "url": länk_url})

            # Skapa DataFrame från den extraherade datan
            df = pd.DataFrame(all_data)
            return df.drop_duplicates()

        except requests.exceptions.RequestException as e:
            return f"Fel vid hämtning av sidan: {e}"
        except Exception as e:
            return f"Ett oväntat fel uppstod: {e}"
    
    list_of_dfs =[]
    for market in markets.items():
        #print (market[0])
        for market_id in market[1]:
            #print (market_id)
            for sector in sectors.items():
                scrape_url = f'https://www.di.se/bors/aktier/?data%5Bcountry%5D=SE&data%5Bmarket%5D={market_id}&data%5Bsector%5D={sector[1]}&field=name&tab=0'
                
                scraped_table=skrapa_uppdelad_tabell(scrape_url)
                print(f'{market[0]}: , Branch: {sector[0]}; {len(scraped_table)} tickers')
                if len(scraped_table) > 0:
                    list_names = scraped_table['text']
                    list_links = scraped_table['url']
                    tickers = get_tickers_from_list(list_links)
                    tickers_uppercase = [item.upper() for item in tickers]
                    df = pd.DataFrame(data = {'Instrument': tickers_uppercase, 'Name': list_names,'Sektor': sector[0]})
                    df['Lista'] = market[0]

                    list_of_dfs.append(df)

                    time.sleep(1)
    df_tickers = pd.concat(list_of_dfs, ignore_index=True) 
    df_tickers = df_tickers[~df_tickers['Instrument'].str.contains('-TO|-BTA|-TR')]

    df_tickers.to_csv(path_file_name, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    config = load_config("rank-config.yaml")
    if config:
        CSV_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')
        TICKERS_FILE_NAME = config["input_ticker_file"]
        if not TICKERS_FILE_NAME:
            print("No tickers file name found. Please check your CSV file.")
        else:
            Create_df_tickers(CSV_PATH/TICKERS_FILE_NAME)
            print("Tickers skrapning klar")
