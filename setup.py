from setuptools import setup, find_packages

setup(
    name='makeatrend',
    version='1.0.0',
    description='Interactive CSV signal viewer with zoom, cursors and custom styling.',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'altgraph==0.17.4',
        'numpy==2.2.4',
        'packaging==24.2',
        'pandas==2.2.3',
        'pefile==2023.2.7',
        'pyinstaller==6.13.0',
        'pyinstaller-hooks-contrib==2025.3',
        'pyqtgraph==0.13.7',
        'PySide6==6.9.0',
        'PySide6_Addons==6.9.0',
        'PySide6_Essentials==6.9.0',
        'python-dateutil==2.9.0.post0',
        'pytz==2025.2',
        'pywin32-ctypes==0.2.3',
        'shiboken6==6.9.0',
        'six==1.17.0',
        'tzdata==2025.2'
    ],
    entry_points={
        'gui_scripts': [
            'makeatrend = main:main'
        ]
    },
    include_package_data=True,
    zip_safe=False
)
