# Socks connection
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent


ua = UserAgent()
proxy = {
        "https": "socks5h://127.0.0.1:9050",
        "http": "socks5h://127.0.0.1:9050"
    }
    # Cookies, User Agent

def crawl_page(url, cookie):
        header = {
            "User-Agent": ua.random,
            "Referer": url
        }

        # Send request
        response = requests.get(url, proxies=proxy,cookies=cookie, headers=header)
        return response

def scrape_products(content):
        soup = BeautifulSoup(content, 'html.parser')
        unique_products = set()
        products = []
        # Products div
        products_div = soup.find('div', {"class" :"Messbar"})
        all_products= products_div.find_all('div', class_='startw')

        for product in all_products:
            # Find the first text node before the <br> tag
            title = product.contents[2].strip()
            
            name = product.find('span', style="color: #F74A3F").text.strip()
            
            # Extract price (the <b> tag with style color #B74A3F)
            price = product.find('span', style="color: #B74A3F").text.strip()
            
            # Extract origin (inside <div class="product-sale-Escrow"> or <div class="product-sale-FE">)
            origin_div = soup.find('div', class_=['product-sale-Escrow', 'product-sale-FE'])
            origin = origin_div.text.strip() if origin_div else "Origin not found"

            product_tuple = (title, origin, name, price)
            if product_tuple not in unique_products:
                        unique_products.add(product_tuple)
                        products.append({
                            'name': title,
                            'destination': origin,
                            'seller': name,
                            'price': price
                        })
        return products


def crawler_exec(version):
    if version == 1:
        url = "http://elysiumpir7gp5j7pl3kye36xnj5zig4fe5ng67ji7s5kahj3jiitfqd.onion/category/289"
    else:
        url= "http://elysiumpir7gp5j7pl3kye36xnj5zig4fe5ng67ji7s5kahj3jiitfqd.onion/category/289?page="+str(version)
    
    cookies = {"elysium_session":"eyJpdiI6IkkzalVlTk8zQWM1K09GS1dPY3AxY0E9PSIsInZhbHVlIjoiU1JrMlFHTlExRklkd0tsK2hZN29GUVZLQUVOcnB0ZllUWTVzTHZZaG8wQzkvYXhaekVZU1NwZGxrK3F3Z3F0bTdVZFRYTUUxS3Z3ZFZOVytTcnpxRnBiZS8zeW50WERlV2FtZWxnaTJqOUVXdU1NMzJWNmJpY2JBWEU0dG5iNWQiLCJtYWMiOiJjMTk3MTIxZGZjMjE1ZTllYTY2ZTYwYjk1MTM0YjA2MjQxN2IzMmJiMWM3ODI1MzJhODFmYjk0YmQzZGRiNmNkIiwidGFnIjoiIn0%3D",
            "XSRF-TOKEN":"eyJpdiI6IkR2R2YzNEhwemkzaEFSS1FnT0t3b1E9PSIsInZhbHVlIjoiVGJaanMwOFBlMU1HYXRZRXBkQUg3cWZ6ejgwQmRka3VLcHZyUno3clNObDdWcjNSVmlRS0gzdTF3Rjl1YTNUZHBmOHM5VXVCQU9RZFF5bi8wMHh1dzFjOUdYNnlWMW1LQWFYZ3RqL0FySnFySHY0TGJtOWhBWXhxMDFaK2NuM00iLCJtYWMiOiI1NjY1MzU1NWEwMDJjYzQxMDIyYzExYjJlNDU1NWVhODhmYmJlNzUzZTFlMDc2NzBhYmNjZjIzNWU5YjM4OGRkIiwidGFnIjoiIn0%3D"              
              }
    
    
    content = crawl_page(url, cookies)
    html_content = content.text
    html_path = "elysium"+str(version)+ ".html"

    # Save html page
    with open(html_path, 'w') as html_file:
        html_file.write(html_content)
        print("wrote to html")

def scraper_exec(version):
    # Load file
    html_path = "Sites/elysium"+str(version)+".html"
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as html_file:
        html_content = html_file.read()

    products = scrape_products(html_content)
    return products

def extract_number_and_check_kg(name):
    # Use regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', name)
    
    if numbers:
        number = int(numbers[0])  # Get the first number found
        # Check if 'kg' is in the string and multiply by 1000 if true
        if 'kg' in name:
            number *= 1000
        return number
    return None  # Return None if no number is found
if __name__ == '__main__':
    #print("crawl pages")
    for i in range(1,20):
       crawler_exec(i)
    all_drugs=[]
    for j in range(1,20):
        new_products=scraper_exec(j)
        print(len(new_products))
        all_drugs.append(new_products)
    # Create DataFrame
    # Flatten the list of lists
    flattened_data = [item for sublist in all_drugs for item in sublist]

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    #df['amount_of_drugs'] = df['name'].apply(extract_number_and_check_kg)
    df.to_csv('elysium.csv')
    


    
