## LIBRARY IMPORTS
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# from matplotlib import pyplot as plt
# import seaborn as sns

# import dash as dash
# from dash import dash_table
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
# from dash.exceptions import PreventUpdate
# import dash_bootstrap_components as dbc
#
import plotly as ply
import plotly.express as px
#
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
#
# import scipy.stats as stats
# import statistics

pd.set_option('display.max_colwidth', 200)


## DATA IMPORTS
INVESTORS_PATH = '/Users/nehat312/GitHub/command-center/prop-swap/data/investors.xlsx'
# MODEL_PATH = r"/Users/nehat312/dsir-426/assignments/projects/capstone/pickle/pickle.pkl"


## VARIABLE ASSIGNMENT
all_investor_idx = pd.read_excel(INVESTORS_PATH, sheet_name = 'PROPSWAP', header = 0)
all_investor_idx = all_investor_idx.sort_values(by = 'TTL VOL RANK')

st.container()
left_column, right_column = st.columns(2)
left_button = left_column.button('PROP/SWAP')
right_button = right_column.button('UNIVERSE')
if left_button:
    left_column.write('*ERROR: CURRENTLY UNDER MAINTENANCE*')
if right_button:
### CLEAN LINK ###
    #st.write('REAL ESTATE INVESTOR UNIVERSE:')
    right_column.write('https://public.tableau.com/shared/D2RKKDK8B?:display_count=n&:origin=viz_share_link')

st.title('PROP/SWAP')
st.header('*VIRTUAL CRE BROKER*')

#st.spinner()
#with st.spinner(text='CONNECTING'):
#    time.sleep(5)
#    st.success('LIVE')

prop_params_header = st.subheader('PROPERTY PARAMETERS:')

sector = st.selectbox(
    '*PROPERTY TYPE:',
    ("MULTIFAMILY",
     "STRIP CENTER", "NNN RETAIL", "MALL",
     "SELF-STORAGE",
     "INDUSTRIAL",
     "FULL-SERVICE HOTEL", "LIMITED-SERVICE HOTEL", "CBD OFFICE", "SUBURBAN OFFICE"))

with st.form("PROPERTY PARAMETERS"):
    if sector == "MULTIFAMILY":
        prop_size = st.slider('*TOTAL MF UNITS: [25-1,000 UNITS]', min_value = 0, max_value = 1000, step = 25)
        #prop_size = st.selectbox('*TOTAL MF UNITS: [25-1,000 UNITS]', list(range(25,750,25)))
    if sector == "FULL-SERVICE HOTEL":
        prop_size = st.selectbox('*TOTAL FS KEYS: [25-1,000 KEYS]', list(range(25,750,25)))
    if sector == "LIMITED-SERVICE HOTEL":
        prop_size = st.selectbox('*TOTAL LS KEYS: [25-1,000 KEYS]', list(range(25,750,25)))
    if sector == "STRIP CENTER":
        prop_size = st.selectbox('*TOTAL SC SF: [5K-1MM SF]', list(range(5000,1005000,5000)))
    if sector == "NNN RETAIL":
        prop_size = st.selectbox('*TOTAL NNN SF: [5K-500k SF]', list(range(5000,505000,5000)))
    if sector == "MALL":
        prop_size = st.selectbox('*TOTAL MALL SF: [10K-1MM SF]', list(range(10000,1010000,10000)))
    if sector == "SELF-STORAGE":
        prop_size = st.selectbox('*TOTAL SELF-STORAGE SF: [5K-500K SF]', list(range(0,525000,25000)))
    if sector == "INDUSTRIAL":
        prop_size = st.selectbox('*TOTAL INDUSTRIAL SF: [5K-1MM SF]', list(range(5000,1005000,5000)))
    if sector == "CBD OFFICE":
        prop_size = st.selectbox('*TOTAL CBD OFFICE SF: [10K-500K SF]', list(range(10000,505000,5000)))
    if sector == "SUBURBAN OFFICE":
        prop_size = st.selectbox('*TOTAL SUB OFFICE SF: [10K-500K SF]', list(range(10000,505000,5000)))

#streamlit. slider ( label , min_value=None , max_value=None , value=None , step=None , format=None , key=None )

    min_prop_price = st.slider('*MINIMUM SALE PRICE [$0MM-$100MM]:', min_value = 0, max_value = 100, step = 5)
        #min_prop_price = st.selectbox('*MINIMUM PRICE [$0MM-$100MM]:', (list(range(0,105,5))))

    #sector = st.selectbox('*PROPERTY REGION:', ("NORTHEAST", "MID-ATLANTIC", "SOUTHEAST", "WEST", "NORTHWEST", "MIDWEST", "SOUTHWEST"))

    prop_qual = st.selectbox(
    '*PROPERTY QUALITY [1-5]:',
    list(range(1,6,1)))

    if min_prop_price == 0:
        st.write('')
    elif min_prop_price > 0:
        implied_ppu_title = st.write('*IMPLIED VALUE / UNIT:')
        implied_ppu = st.markdown(round(min_prop_price * 1_000_000 / prop_size))

    params_submit = st.form_submit_button("PROP/SWAP")

### PICKLE PICKLE PICKLE ###

    investor_cols = ['INVESTOR', 'INVESTOR TYPE', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'C-SUITE']
    mf_cols = ['INVESTOR', 'MF AVG PRICE ($M)', 'MF UNITS / PROP', 'MF AVG PPU',  'AVG QUALITY', 'MF QUALITY', 'TTL VOL RANK', 'TTL SF RANK', 'MF VOL RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'INVESTOR TYPE']
    sc_cols = ['INVESTOR', 'SC AVG PRICE ($M)', 'SC SF / PROP', 'SC AVG PSF',  'AVG QUALITY', 'SC QUALITY', 'TTL VOL RANK', 'TTL SF RANK', 'SC VOL RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'INVESTOR TYPE']
    nnn_cols = ['INVESTOR', 'NNN AVG PRICE ($M)', 'NNN SF / PROP', 'NNN AVG PSF',  'AVG QUALITY', 'NNN QUALITY', 'TTL VOL RANK', 'TTL SF RANK', 'NNN VOL RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'INVESTOR TYPE']
    mall_cols = ['INVESTOR', 'MALL AVG PRICE ($M)', 'MALL SF / PROP', 'MALL AVG PSF',  'AVG QUALITY', 'MALL QUALITY', 'TTL VOL RANK', 'TTL SF RANK', 'MALL VOL RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'INVESTOR TYPE']
    ss_cols = ['INVESTOR', 'SS AVG PRICE ($M)', 'SS SF / PROP',  'SS AVG PSF',  'AVG QUALITY', 'SS QUALITY', 'TTL VOL RANK', 'TTL SF RANK', 'SS VOL RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'INVESTOR TYPE']
    ind_cols = ['INVESTOR', 'IND AVG PRICE ($M)', 'IND SF / PROP', 'IND AVG PSF',  'AVG QUALITY', 'IND QUALITY', 'TTL VOL RANK', 'TTL SF RANK', 'IND VOL RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'INVESTOR TYPE']
    fs_cols = ['INVESTOR', 'FS AVG PRICE ($M)', 'FS KEYS / PROP', 'FS AVG PPK',  'AVG QUALITY', 'FS QUALITY', 'TTL VOL RANK', 'TTL SF RANK', 'FS VOL RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'INVESTOR TYPE']
    ls_cols = ['INVESTOR', 'LS AVG PRICE ($M)', 'LS KEYS / PROP', 'LS AVG PPK',  'AVG QUALITY', 'LS QUALITY', 'TTL VOL RANK', 'TTL SF RANK', 'LS VOL RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'INVESTOR TYPE']
    cbd_cols = ['INVESTOR', 'CBD AVG PRICE ($M)', 'CBD SF / PROP', 'CBD AVG PSF',  'AVG QUALITY', 'CBD QUALITY', 'TTL VOL RANK', 'TTL SF RANK', 'CBD VOL RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'INVESTOR TYPE']
    sub_cols = ['INVESTOR', 'SUB AVG PRICE ($M)', 'SUB SF / PROP', 'SUB AVG PSF',  'AVG QUALITY', 'SUB QUALITY', 'TTL VOL RANK', 'TTL SF RANK', 'SUB VOL RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'INVESTOR TYPE']

    @st.cache(persist = True, allow_output_mutation = True)
    def filter_buyers(sector, prop_size, min_prop_price, prop_qual):
      if sector == 'MULTIFAMILY':
        for investors in all_investor_idx:
          mf_size_filter = all_investor_idx[all_investor_idx['MF UNITS / PROP'] >= prop_size]
          mf_min_price_filter = mf_size_filter[mf_size_filter['MF AVG PRICE ($M)'] >= min_prop_price]
          mf_qual_filter = mf_min_price_filter[(mf_min_price_filter['MF QUALITY'] >= (prop_qual-1)) & (mf_min_price_filter['MF QUALITY'] <= (prop_qual+1))]
          mf_buyer_recs = mf_qual_filter.sort_values(by = 'MF VOL RANK', ascending = True)[:50]
          mf_buyer_recs = pd.DataFrame(data = mf_buyer_recs, columns = mf_cols)
        return mf_buyer_recs
      elif sector == 'STRIP CENTER':
        for investors in all_investor_idx:
          sc_size_filter = all_investor_idx[all_investor_idx['SC SF / PROP'] >= prop_size]
          sc_min_price_filter = sc_size_filter[sc_size_filter['SC AVG PRICE ($M)'] >= min_prop_price]
          sc_qual_filter = sc_min_price_filter[(sc_min_price_filter['SC QUALITY'] >= (prop_qual-1)) & (sc_min_price_filter['SC QUALITY'] <= (prop_qual+1))]
          sc_buyer_recs = sc_qual_filter.sort_values(by = 'SC VOL RANK', ascending = True)[:20]
          sc_buyer_recs = pd.DataFrame(data = sc_buyer_recs, columns = sc_cols)
        return sc_buyer_recs
      elif sector == 'NNN RETAIL':
        for investors in all_investor_idx:
          nnn_size_filter = all_investor_idx[all_investor_idx['NNN SF / PROP'] >= prop_size]
          nnn_min_price_filter = nnn_size_filter[nnn_size_filter['NNN AVG PRICE ($M)'] >= min_prop_price]
          nnn_qual_filter = nnn_min_price_filter[(nnn_min_price_filter['NNN QUALITY'] >= (prop_qual-1)) & (nnn_min_price_filter['NNN QUALITY'] <= (prop_qual+1))]
          nnn_buyer_recs = nnn_qual_filter.sort_values(by = 'NNN VOL RANK', ascending = True)[:50]
          nnn_buyer_recs = pd.DataFrame(data = nnn_buyer_recs, columns = nnn_cols)
        return nnn_buyer_recs
      elif sector == 'MALL':
        for investors in all_investor_idx:
          mall_size_filter = all_investor_idx[all_investor_idx['MALL SF / PROP'] >= prop_size]
          mall_min_price_filter = mall_size_filter[mall_size_filter['MALL AVG PRICE ($M)'] >= min_prop_price]
          mall_qual_filter = mall_min_price_filter[(mall_min_price_filter['MALL QUALITY'] >= (prop_qual-2)) & (mall_min_price_filter['MALL QUALITY'] <= (prop_qual+2))]
          mall_buyer_recs = mall_qual_filter.sort_values(by = 'MALL VOL RANK', ascending = False)[:12]
          mall_buyer_recs = pd.DataFrame(data = mall_buyer_recs, columns = mall_cols)
        return mall_buyer_recs
      elif sector == 'SELF-STORAGE':
        for investors in all_investor_idx:
          ss_size_filter = all_investor_idx[all_investor_idx['SS SF / PROP'] >= prop_size]
          ss_min_price_filter = ss_size_filter[ss_size_filter['SS AVG PRICE ($M)'] >= min_prop_price]
          ss_qual_filter = ss_min_price_filter[(ss_min_price_filter['SS QUALITY'] >= (prop_qual-1)) & (ss_min_price_filter['SS QUALITY'] <= (prop_qual+1))]
          ss_buyer_recs = ss_qual_filter.sort_values(by = 'SS VOL RANK', ascending = True)[:50]
          ss_buyer_recs = pd.DataFrame(data = ss_buyer_recs, columns = ss_cols)
        return ss_buyer_recs
      elif sector == 'INDUSTRIAL':
        for investors in all_investor_idx:
          ind_size_filter = all_investor_idx[all_investor_idx['IND SF / PROP'] >= prop_size]
          ind_min_price_filter = ind_size_filter[ind_size_filter['IND AVG PRICE ($M)'] >= min_prop_price]
          ind_qual_filter = ind_min_price_filter[(ind_min_price_filter['IND QUALITY'] >= (prop_qual-1)) & (ind_min_price_filter['IND QUALITY'] <= (prop_qual+1))]
          ind_buyer_recs = ind_qual_filter.sort_values(by = 'IND VOL RANK', ascending = True)[:50]
          ind_buyer_recs = pd.DataFrame(data = ind_buyer_recs, columns = ind_cols)
        return ind_buyer_recs
      elif sector == 'FULL-SERVICE HOTEL':
        for investors in all_investor_idx:
          fs_size_filter = all_investor_idx[all_investor_idx['FS KEYS / PROP'] >= prop_size]
          fs_min_price_filter = fs_size_filter[fs_size_filter['FS AVG PRICE ($M)'] >= min_prop_price]
          fs_qual_filter = fs_min_price_filter[(fs_min_price_filter['FS QUALITY'] >= (prop_qual-1)) & (fs_min_price_filter['FS QUALITY'] <= (prop_qual+1))]
          fs_buyer_recs = fs_qual_filter.sort_values(by = 'FS VOL RANK', ascending = True)[:50]
          fs_buyer_recs = pd.DataFrame(data = fs_buyer_recs, columns = fs_cols)
        return fs_buyer_recs
      elif sector == 'LIMITED-SERVICE HOTEL':
        for investors in all_investor_idx:
          ls_size_filter = all_investor_idx[all_investor_idx['LS KEYS / PROP'] >= prop_size]
          ls_min_price_filter = ls_size_filter[ls_size_filter['LS AVG PRICE ($M)'] >= min_prop_price]
          ls_qual_filter = ls_min_price_filter[(ls_min_price_filter['LS QUALITY'] >= (prop_qual-1)) & (ls_min_price_filter['LS QUALITY'] <= (prop_qual+1))]
          ls_buyer_recs = ls_qual_filter.sort_values(by = 'LS VOL RANK', ascending = True)[:50]
          ls_buyer_recs = pd.DataFrame(data = ls_buyer_recs, columns = ls_cols)
        return ls_buyer_recs
      elif sector == 'CBD OFFICE':
        for investors in all_investor_idx:
          cbd_size_filter = all_investor_idx[all_investor_idx['CBD SF / PROP'] >= prop_size]
          cbd_min_price_filter = cbd_size_filter[cbd_size_filter['CBD AVG PRICE ($M)'] >= min_prop_price]
          cbd_qual_filter = cbd_min_price_filter[(cbd_min_price_filter['CBD QUALITY'] >= (prop_qual-1)) & (cbd_min_price_filter['CBD QUALITY'] <= (prop_qual+1))]
          cbd_buyer_recs = cbd_qual_filter.sort_values(by = 'CBD VOL RANK', ascending = True)[:50]
          cbd_buyer_recs = pd.DataFrame(data = cbd_buyer_recs, columns = cbd_cols)
        return cbd_buyer_recs
      elif sector == 'SUB OFFICE':
        for investors in all_investor_idx:
          sub_size_filter = all_investor_idx[all_investor_idx['SUB SF / PROP'] >= prop_size]
          sub_min_price_filter = sub_size_filter[sub_size_filter['SUB AVG PRICE ($M)'] >= min_prop_price]
          sub_qual_filter = sub_min_price_filter[(sub_min_price_filter['SUB QUALITY'] >= (prop_qual-1)) & (sub_min_price_filter['SUB QUALITY'] <= (prop_qual+1))]
          sub_buyer_recs = sub_qual_filter.sort_values(by = 'SUB VOL RANK', ascending = True)[:50]
          sub_buyer_recs = pd.DataFrame(data = sub_buyer_recs, columns = sub_cols)
        return sub_buyer_recs

## INVESTOR RECOMMENDATIONS ##
    if params_submit:
        st.write("RECOMMENDED INVESTOR POOL:")
        buyer_rec_df = filter_buyers(sector, prop_size, min_prop_price, prop_qual)
        buyer_rec_df.set_index('INVESTOR', inplace = True)
        st.dataframe(buyer_rec_df)

        if sector == 'MULTIFAMILY':
            per_unit_valuation = round(buyer_rec_df['MF AVG PPU'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED PROPERTY VALUE / UNIT:")
            st.write(per_unit_valuation)
            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.barplot(y = buyer_rec_df['INVESTOR TYPE'], x = buyer_rec_df['MF AVG PPU'], palette = 'mako', ci = None, orient = 'h')
            # plt.xlabel('AVG MULTIFAMILY PPU', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)
        elif sector == 'STRIP CENTER':
            per_unit_valuation = round(buyer_rec_df['SC AVG PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)
            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.barplot(y = buyer_rec_df['INVESTOR TYPE'], x = buyer_rec_df['SC AVG PSF'], palette = 'mako', ci = None, orient = 'h')
            # plt.xlabel('AVG STRIP CENTER VALUE PSF', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)
        elif sector == 'NNN RETAIL':
            per_unit_valuation = round(buyer_rec_df['NNN AVG PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)
            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.boxplot(y = buyer_rec_df['INVESTOR TYPE'], x = buyer_rec_df['NNN AVG PSF'], palette = 'mako', orient = 'h')
            # #plt.title('AVERAGE RETAIL PRICE PSF', fontsize = 24)
            # plt.xlabel('AVG NNN RETAIL PRICE ($MM)', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)
        elif sector == 'MALL':
            per_unit_valuation = round(buyer_rec_df['MALL AVG PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)
            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.barplot(y = buyer_rec_df['INVESTOR TYPE'], x = buyer_rec_df['MALL AVG PSF'], palette = 'mako', ci = None, orient = 'h')
            # #plt.title('AVERAGE RETAIL PRICE PSF', fontsize = 24)
            # plt.xlabel('AVG MALL PRICE ($MM)', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)
        elif sector == 'SELF-STORAGE':
            per_unit_valuation = round(buyer_rec_df['SS AVG PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)
            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.barplot(y = buyer_rec_df['INVESTOR TYPE'], x = buyer_rec_df['SS AVG PRICE ($M)'], palette = 'mako', ci = None, orient = 'h')
            # #plt.title('AVERAGE RETAIL PRICE PSF', fontsize = 24)
            # plt.xlabel('AVG SELF-STORAGE PRICE ($MM)', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)
        elif sector == 'INDUSTRIAL':
            per_unit_valuation = round(buyer_rec_df['IND AVG PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)
            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.barplot(y = buyer_rec_df['INVESTOR TYPE'], x = buyer_rec_df['IND AVG PRICE ($M)'], palette = 'mako', ci = None, orient = 'h')
            # #plt.title('AVERAGE RETAIL PRICE PSF', fontsize = 24)
            # plt.xlabel('AVG INDUSTRIAL PRICE ($MM)', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)
        elif sector == 'FULL-SERVICE HOTEL':
            per_unit_valuation = round(buyer_rec_df['FS AVG PPK'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE / KEY:")
            st.write(per_unit_valuation)
            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.barplot(y = buyer_rec_df['INVESTOR TYPE'], x = buyer_rec_df['FS AVG PRICE ($M)'], palette = 'mako', ci = None, orient = 'h')
            # #plt.title('AVERAGE RETAIL PRICE PSF', fontsize = 24)
            # plt.xlabel('AVG FS HOTEL PRICE ($MM)', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)
        elif sector == 'LIMITED-SERVICE HOTEL':
            per_unit_valuation = round(buyer_rec_df['LS AVG PPK'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE / KEY:")
            st.write(per_unit_valuation)
            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.barplot(y = buyer_rec_df['INVESTOR TYPE'], x = buyer_rec_df['LS AVG PRICE ($M)'], palette = 'mako', ci = None, orient = 'h')
            # #plt.title('AVERAGE RETAIL PRICE PSF', fontsize = 24)
            # plt.xlabel('AVG LS HOTEL PRICE ($MM)', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)
        elif sector == 'CBD OFFICE':
            per_unit_valuation = round(buyer_rec_df['CBD AVG PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)
            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.barplot(y = buyer_rec_df['INVESTOR TYPE'], x = buyer_rec_df['CBD AVG PRICE ($M)'], palette = 'mako', ci = None, orient = 'h')
            # #plt.title('AVERAGE RETAIL PRICE PSF', fontsize = 24)
            # plt.xlabel('AVG CBD OFFICE PRICE ($MM)', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)
        elif sector == 'SUB OFFICE':
            per_unit_valuation = round(buyer_rec_df['SUB AVG PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)
            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.barplot(y = buyer_rec_df['INVESTOR TYPE'], x = buyer_rec_df['SUB AVG PRICE ($M)'], palette = 'mako', ci = None, orient = 'h')
            # #plt.title('AVG SUB OFFICE PRICE', fontsize = 24)
            # plt.xlabel('AVG SUB OFFICE PRICE ($MM)', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)

### EXPLAIN QUALITY SCALE ###

## CREDITS / FOOTNOTES
st.success('THANKS FOR PROP/SWAPPING')
    #st.warning('NO BUYERS FOUND')
st.write('*~PROP/SWAP BETA MODE~*')
st.stop()