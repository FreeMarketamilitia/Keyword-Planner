from setuptools import setup, find_packages

setup(
    name='Keyword-Kraken',  # Name of the package
    version='1.0',  # Version of the package
    description='Keyword Kraken: A powerful SEO tool for keyword analysis, SERP tracking, and trend predictions, with a touch of Kraken-powered magic.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='FreeMarketMilitia',
    author_email='freemarket@nostates.com',
    url='https://github.com/FreeMarketamilitia/Keyword-Planner', 
    packages=find_packages(),  # Automatically find all packages in the directory
    install_requires=[  # External dependencies
        'Flask==2.2.3',
        'pandas==1.5.3',
        'google-auth==2.16.0',
        'google-auth-oauthlib==0.8.0',
        'google-api-python-client==2.87.0',
        'google-ads==20.0.0',
        'google-generativeai==0.2.0',
        'python-dotenv==0.21.1',
        'tenacity==8.0.1',
        'nltk==3.7',
        'scikit-learn==1.1.2',
        'requests==2.28.2',
        'beautifulsoup4==4.11.1',
    ],
    classifiers=[  # Metadata about the package
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Minimum Python version
)
