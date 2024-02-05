import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta, date
from pathlib import Path
import os
from os import walk
from dateutil.parser import parse
import locale

def get_src_directory(script_directory):
    src_directory = script_directory.parent / 'src'
    return src_directory

#excel_tab = get_src_directory

filenames = next(walk('/Users/danielzellner/Documents/Studium/Bachelorthesis/src'), (None, None, []))[2]
print(filenames)