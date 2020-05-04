from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from selenium.common.exceptions import TimeoutException, WebDriverException

import re
from pathlib import Path
import os

def set_selenium_chrome_session(use_proxy=True,load_without_images=True):
    chrome_options = webdriver.ChromeOptions()

    if use_proxy:

        home = str(Path.home())
        tor_executable_path = home + r'\Desktop\Tor Browser\Browser\TorBrowser\Tor\tor.exe'
        os.popen(tor_executable_path) #Si el navegador no carga, validar instalación de Tor
        PROXY = "socks5://localhost:9050" # IP:PORT or HOST:PORT
        chrome_options.add_argument('--proxy-server=%s' % PROXY)
    
    if load_without_images:
        chrome_prefs = {}
        chrome_options.experimental_options["prefs"] = chrome_prefs
        chrome_prefs["profile.default_content_settings"] = {"images": 2}
        chrome_prefs["profile.managed_default_content_settings"] = {"images": 2}

    return webdriver.Chrome("chromedriver_win32/chromedriver",options=chrome_options)

def wait_for_clickable_element(driver,element_id_string,error_message,timeout=20):

    try:
        return WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.ID, element_id_string)))

    except TimeoutException:
        print(error_message)
        driver.quit()

def wait_for_text_in_element(driver,element_id_string,desired_string,error_message,timeout=20):

    try:
        return WebDriverWait(driver, timeout).until(EC.text_to_be_present_in_element((By.CLASS_NAME, element_id_string),desired_string))

    except TimeoutException:
        print(error_message)
        driver.quit()


def send_keys_and_enter(clickable_object,key_string):
    clickable_object.send_keys(key_string)
    clickable_object.send_keys(Keys.ENTER)
