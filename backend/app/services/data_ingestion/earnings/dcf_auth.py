"""
Discounting Cash Flows (DCF) Authentication Module

Handles login and session management for discountingcashflows.com
"""

import requests
import logging
import os
from typing import Optional
from pathlib import Path
import pickle
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class DCFAuthenticator:
    """Authenticator for discountingcashflows.com"""

    BASE_URL = "https://discountingcashflows.com"
    LOGIN_URL = f"{BASE_URL}/accounts/login/"

    def __init__(self, email: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize authenticator

        Args:
            email: Login email (or set DCF_EMAIL env var)
            password: Login password (or set DCF_PASSWORD env var)
        """
        self.email = email or os.getenv('DCF_EMAIL')
        self.password = password or os.getenv('DCF_PASSWORD')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
        })
        self._cookies_file = Path.home() / '.dcf_cookies.pkl'
        self._is_authenticated = False

    def login(self, force_new: bool = False) -> bool:
        """
        Login to discountingcashflows.com

        Args:
            force_new: Force new login even if cached session exists

        Returns:
            True if login successful, False otherwise
        """
        if not force_new and self._load_session():
            logger.info("Using cached session")
            return True

        if not self.email or not self.password:
            logger.error("Email and password required for login")
            return False

        logger.info(f"Logging in to {self.BASE_URL}")

        try:
            # Step 1: Get login page to retrieve CSRF token
            response = self.session.get(self.LOGIN_URL, timeout=30)
            response.raise_for_status()

            # Parse CSRF token
            soup = BeautifulSoup(response.text, 'html.parser')
            csrf_token = self._extract_csrf_token(soup)

            if not csrf_token:
                logger.error("Could not find CSRF token on login page")
                return False

            logger.debug(f"CSRF token: {csrf_token[:10]}...")

            # Step 2: Submit login form
            login_data = {
                'csrfmiddlewaretoken': csrf_token,
                'login': self.email,
                'password': self.password,
            }

            headers = {
                'Referer': self.LOGIN_URL,
                'Origin': self.BASE_URL,
            }

            response = self.session.post(
                self.LOGIN_URL,
                data=login_data,
                headers=headers,
                timeout=30,
                allow_redirects=True
            )

            # Check if login was successful
            if self._check_login_success(response):
                logger.info("Login successful")
                self._is_authenticated = True
                self._save_session()
                return True
            else:
                logger.error("Login failed - check credentials")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Login request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during login: {e}")
            return False

    def _extract_csrf_token(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract CSRF token from login page"""
        # Method 1: Look for hidden input
        csrf_input = soup.find('input', {'name': 'csrfmiddlewaretoken'})
        if csrf_input and csrf_input.get('value'):
            return csrf_input['value']

        # Method 2: Look in meta tags
        csrf_meta = soup.find('meta', {'name': 'csrf-token'})
        if csrf_meta and csrf_meta.get('content'):
            return csrf_meta['content']

        # Method 3: Check cookies
        if 'csrftoken' in self.session.cookies:
            return self.session.cookies['csrftoken']

        return None

    def _check_login_success(self, response: requests.Response) -> bool:
        """
        Check if login was successful

        Args:
            response: Response from login POST

        Returns:
            True if login successful
        """
        # Method 1: Check if redirected away from login page
        if '/accounts/login' not in response.url:
            return True

        # Method 2: Check for error messages
        soup = BeautifulSoup(response.text, 'html.parser')
        error_divs = soup.find_all(['div', 'p'], class_=lambda x: x and 'error' in x.lower())
        if error_divs:
            return False

        # Method 3: Check for login form still present
        login_form = soup.find('form', {'method': 'post'})
        if login_form and 'login' in str(login_form).lower():
            return False

        # Method 4: Check for session cookies
        if 'sessionid' in self.session.cookies:
            return True

        return False

    def _save_session(self):
        """Save session cookies to file"""
        try:
            with open(self._cookies_file, 'wb') as f:
                pickle.dump(self.session.cookies, f)
            logger.debug(f"Session saved to {self._cookies_file}")
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

    def _load_session(self) -> bool:
        """
        Load session cookies from file

        Returns:
            True if session loaded and valid
        """
        if not self._cookies_file.exists():
            return False

        try:
            with open(self._cookies_file, 'rb') as f:
                cookies = pickle.load(f)
                self.session.cookies.update(cookies)

            # Verify session is still valid
            if self._verify_session():
                self._is_authenticated = True
                return True
            else:
                logger.debug("Cached session expired")
                return False

        except Exception as e:
            logger.debug(f"Failed to load session: {e}")
            return False

    def _verify_session(self) -> bool:
        """
        Verify that current session is authenticated

        Returns:
            True if session is valid
        """
        try:
            # Try accessing a page that requires authentication
            test_url = f"{self.BASE_URL}/company/AAPL/transcripts/2024/4/"
            response = self.session.get(test_url, timeout=10)

            # Check if we're redirected to login
            if '/accounts/login' in response.url:
                return False

            # Check if content is accessible (not showing login prompt)
            soup = BeautifulSoup(response.text, 'html.parser')
            login_link = soup.find('a', href='/accounts/login/')

            # If login link is present, we're not logged in
            if login_link and 'Login' in login_link.get_text():
                return False

            return True

        except:
            return False

    def get_authenticated_session(self) -> Optional[requests.Session]:
        """
        Get authenticated session

        Returns:
            Authenticated session or None if login fails
        """
        if not self._is_authenticated:
            if not self.login():
                return None

        return self.session

    def logout(self):
        """Logout and clear cached session"""
        if self._cookies_file.exists():
            self._cookies_file.unlink()
        self._is_authenticated = False
        self.session.cookies.clear()
        logger.info("Logged out and cleared session")


# Global authenticator instance
dcf_authenticator = DCFAuthenticator()
