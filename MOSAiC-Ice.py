#####################################################


import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pydeck as pdk
from PIL import Image
import requests
from io import BytesIO
import base64

def _max_width_():
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


class Buoy:
    def __init__(self, buoyid, temp, heat, loc, startdate,rawdata):


        self.buoyid=buoyid
        self.temp=temp
        self.heat=heat
        self.loc=loc
        self.startdate=startdate
        self.url=rawdata
         
        

    #@st.cache    
    def processbuoy(self):
        st.sidebar.title("Help improve the model")
        st.sidebar.markdown(" Data will not be cached and will take time to process")

        #st.sidebar.header("Th(snow)")
        #st.sidebar.markdown("Tweak our parameter for calculating Th(ice).")
        snowdelta= st.sidebar.number_input("Tweak Th(snow) by:",min_value=float(-2), max_value=float(2),value=float(0),step=0.1) 

        #st.sidebar.header("Th(ice)")
        #st.sidebar.markdown("Tweak our parameter for calculating Th(ice).")
        delta= st.sidebar.number_input("Tweak Th(ice) by:", min_value=float(-1), max_value=float(1),value=float(0),step=0.1)

        #st.sidebar.header("Th(ocean)")
        #st.sidebar.markdown("Tweak our parameter for calculating Th(ocean)")
        oceandelta= st.sidebar.number_input("Tweak Th(ocean) by:", min_value=float(-3), max_value=float(3),value=float(0),step=0.2)


        @st.cache
        def prepdata():
            
                     
            url_HEAT=self.heat
            df_HEAT_main=pd.read_csv(url_HEAT)

            url_TS=self.loc
            df_TS=pd.read_csv(url_TS)
            df_TS["Date"] = df_TS["time"].str[:10]
            #df_TS.drop_duplicates(subset ="date", keep = "first", inplace = True) 
            df_TS.columns=["ts_time","ts_latitude (deg)","ts_longitude (deg)","Date"]

            url_TEMP=self.temp
            df_TEMP=pd.read_csv(url_TEMP)
            df_TEMP=df_TEMP.iloc[::4, :]

            time_thickness=df_TEMP.iloc[:,:4]
            time_thickness["date"] = time_thickness["time"].str[:10]

            #time_thickness=df_TS.merge(time_thickness, left_on="date", right_on="date")
            #time_thickness=time_thickness.drop(["time", "latitude (deg)","longitude (deg)"], axis=1)
            names= ["Time","Latitude","Longitude","Air_Temp_(C)","Date"]  
            time_thickness.columns=names
            time_thickness["Time"] = time_thickness["Time"].str[11:]
            #time_thickness=time_thickness[["Date","Time","Latitude","Longitude","Air_Temp_(C)"]]




            number_of_days=len(df_HEAT_main)


            daily_ice_thickness=[]
            daily_snow_thickness=[]
            daily_Th_snow=[]
            daily_Th_ice=[]
            daily_Th_ocean=[]
            plots=[]


            i=(-1)

            for value in range (0,(number_of_days-10)):



                df_HEAT=df_HEAT_main.iloc[(i-10):i]
                df_HEAT=df_HEAT.drop(['time', 'latitude (deg)','longitude (deg)'], axis=1)        
                df_HEAT.loc['stdev'] = df_HEAT.std()
                df_HEAT.loc['mean_t'] = df_HEAT.mean()        
                df_HEAT=df_HEAT.tail(2)
                df_HEAT=df_HEAT.T
                df_HEAT.reset_index(level=0, inplace=True)
                df_HEAT["Temp_Change"]=df_HEAT.mean_t.diff()
                df_ocean=df_HEAT.tail(50)
                core_temperature=df_ocean.mean_t.mean()
                df_HEAT=df_HEAT.head(-5)
                df_HEAT["Deltacore"]=df_HEAT.mean_t-core_temperature
                thermistors_not_in_ice=df_HEAT.where(df_HEAT.Deltacore>(1+delta))
                thermistors_not_in_ice=thermistors_not_in_ice.dropna()
                Th_ice= thermistors_not_in_ice.tail(1)
                Th_ice=list(Th_ice.index.values)
                Th_ice=int(Th_ice[0])
                
                if buoyid=='T66':
                    Th_ice=Th_ice-2
                else:
                    pass


                daily_Th_ice.append(Th_ice)



                df_HEAT=df_HEAT_main[(i-150):i]
                df_HEAT=df_HEAT.drop(['time', 'latitude (deg)','longitude (deg)'], axis=1)
                df_HEAT=df_HEAT.iloc[:,:Th_ice]
                df_HEAT = df_HEAT.reset_index(drop=True)
                df_HEAT.loc['stdev'] = df_HEAT.std()
                df_HEAT=df_HEAT.tail(1)
                df_HEAT=df_HEAT.T
                df_air=df_HEAT.head(30)
                airstdev=df_air.stdev.mean()
                df_HEAT["Deltaair"]=df_HEAT.stdev-airstdev
                thermistors_in_snow=df_HEAT.where(df_HEAT.Deltaair<(-0.5+snowdelta))
                thermistors_in_snow=thermistors_in_snow.dropna()
                snow_depth=round(((len(thermistors_in_snow)*2)/100),3)
                Th_snow=Th_ice-(len(thermistors_in_snow))



                daily_Th_snow.append(Th_snow)
                daily_snow_thickness.append(snow_depth)





                df_TEMP_subset=df_TEMP.iloc[(i-10):i]
                df_TEMP_subset=df_TEMP_subset.drop(['time', 'latitude (deg)','longitude (deg)'], axis=1)
                df_TEMP_subset.loc['stdev'] = df_TEMP_subset.std()
                df_TEMP_subset.loc['mean_t'] = df_TEMP_subset.mean()
                df_TEMP_subset=df_TEMP_subset.tail(2)
                df_TEMP_subset=df_TEMP_subset.T
                df_TEMP_subset.reset_index(level=0, inplace=True)
                total_thermistors_in_column=len(df_TEMP_subset)
                ice_or_water_thermistors=df_TEMP_subset.tail(total_thermistors_in_column-Th_ice)
                ice_thermistors=ice_or_water_thermistors.where(ice_or_water_thermistors.mean_t<(-2.2+oceandelta))
                ice_thermistors=ice_thermistors.dropna()
                Th_ocean=ice_thermistors.tail(2)
                Th_ocean=list(Th_ocean.index.values)
                Th_ocean=int(Th_ocean[0])
                ice_thickness=list(ice_thermistors.index.values)
                ice_thickness=(len(ice_thickness)*2)/100



                daily_Th_ocean.append(Th_ocean)
                daily_ice_thickness.append(ice_thickness)




                i-=1



            time_thickness=time_thickness.iloc[::-1]
            time_thickness=time_thickness.reset_index(drop=True)

            Ice_Thickness=pd.DataFrame((daily_ice_thickness),columns=["Ice_Thickness_(m)"])
            Snow_Thickness=pd.DataFrame((daily_snow_thickness),columns=["Snow_Thickness_(m)"])                               
            Th_snow=pd.DataFrame((daily_Th_snow),columns=["Th(snow)"])                            
            Th_ice=pd.DataFrame((daily_Th_ice),columns=["Th(ice)"])   
            Th_ocean=pd.DataFrame((daily_Th_ocean),columns=["Th(ocean)"])

            variables= [Ice_Thickness,Snow_Thickness,Th_snow,Th_ice,Th_ocean]                                   
            for variable in variables:

                time_thickness = pd.concat([time_thickness,variable],axis=1,sort=False)

            time_thickness=time_thickness.dropna()    
            time_thickness=time_thickness.iloc[::-1]
            time_thickness=time_thickness.reset_index(drop=True)
            time_thickness=time_thickness.dropna()        
            time_thickness=time_thickness.sort_values(by=["Date"])

            loc=time_thickness.loc[time_thickness["Date"]==self.startdate].index[0]
            time_thickness=time_thickness.iloc[loc:]
            time_thickness["Change_in_Thickness_(m)"]= time_thickness["Ice_Thickness_(m)"].diff()



            buoy_id=self.buoyid
            time_thickness["Buoy"]=buoy_id
            #time_thickness= pd.DataFrame(np.repeat(time_thickness.values,12,axis=0))
            #newdf = pd.DataFrame(np.repeat(df.values,3,axis=0))

            time_thickness=pd.merge(df_TS,time_thickness,on="Date")


            time_thickness["Ice_Board"]=0
            time_thickness["Ice_Bottom"]=0-time_thickness["Ice_Thickness_(m)"]
            time_thickness["Snow"]=0+time_thickness["Snow_Thickness_(m)"]
            
            return time_thickness



        time_thickness= prepdata()
        

        #ax1.set(xticks=np.arange(min(x), max(x)+1, 1.0))
        
        class Plot:
            def __init__(self,time_thickness_final):
       
                zoom = st.slider('Zoom to Current Date', 0, len(time_thickness), 1)
                zoomed = len(time_thickness)- zoom
                self.yaxis = st.slider('Stretch y-axis', 5, 12, 6)
                #zoomeddf= time_thickness.tail(zoomed)
                self.df=time_thickness_final.tail(zoomed)

                
               
            def createplot(self):
                fig = plt.figure(figsize=(12,self.yaxis))
        
                ax1 = fig.add_subplot(1, 1, 1)  
                plt.plot(self.df.Date, self.df.Ice_Bottom)
                plt.plot(self.df.Date, self.df.Snow,color="grey")

                plt.fill_between(self.df.Date,self.df.Ice_Bottom,-3,color="navy")
                plt.fill_between(self.df.Date,0,self.df.Ice_Bottom,color="azure")

                time_thickness_drops=self.df.drop_duplicates(subset =["Date"], keep = "first")
                datelen=time_thickness_drops.Date.tolist()
                datelenfull=self.df.Date.tolist()
                ax1.set(xticks=range(0, len(datelen), 10))
                ax1.set_ylabel("Delta (Metres)",fontsize=15)
                plt.xticks(rotation=45)
                plt.ylim([-3, 1])
                #plt.xlim([0,140])
                plt.xlim([0,len(datelen)])
                plt.axhline(y=0,color="black",linewidth=0.5)
                plt.title("Buoy: " + str(self.df.Buoy.iloc[1]),fontsize=25,x=0.2)



                ax2 = ax1.twinx() 
                ax2.plot(self.df.Date, self.df["Air_Temp_(C)"],color="red",linewidth=1,ls="dashed")
                #ax2.set(xticks=[0,20, 40, 60, 80, 100, 120, 140])
                ax2.set(xticks=range(0, len(datelen), 10))
                ax2.set_ylabel("Air Temperature (Degrees C)",fontsize=15)

                legend_elements = [Line2D([0], [0], color='grey', lw=3, label='Snow Line'),
                                   Patch(facecolor='azure',edgecolor='black',label='Ice'),
                                   Patch(facecolor='navy', edgecolor='black',label='Ocean'),
                                   Line2D([0], [0], color='red', lw=3, linestyle="--", label='Air Temp')]

                ax1.legend(handles=legend_elements, loc='center',bbox_to_anchor=(1.175, 0.5),labelspacing=2)
                plt.tight_layout()
                #st.header("Seasonal conditions for : " + self.buoyid )
                st.pyplot(fig)

        #st.header("Visualisation")       
        Visualisation=Plot(time_thickness)
        Visualisation.createplot()
        
        st.markdown("Th(snow) determined by difference in standard deviation of specific thermistor versus air-thermistors of: " + str(-0.5+snowdelta)  + "C")
        st.markdown("Th(ice) determined by difference in thermal capacity of specific thermistor versus thermal capacity of ocean-thermistors of:  " + str(1+delta)  + "C")
        st.markdown("Th(ocean) determined by raw temperature of specific thermistor reading below:  " + str(-2.2+oceandelta) + "C")
        

        st.header("What the team at MOSAiC produce (raw-data)")
        
        
        if st.button("Display raw data"):
            response = requests.get(self.url)
            img = Image.open(BytesIO(response.content))
            st.image(img,use_column_width=True)
        
        #zoom = st.slider('Zoom to present', 0, len(time_thickness), 1)
        #zoomed = len(time_thickness)- zoom
        #zoomeddf= time_thickness.tail(zoomed)
        #Visualisation2=Plot(zoomeddf)
        
                
                
                
        st.header("Analysed Data")
 
        
        st.markdown("Hit top right of dataframe to enlarge")

        dftype = st.radio("Toggle dataset type:",('Full Dataset', 'Key Metrics Only'))
        if dftype == 'Full Dataset':
            st.dataframe(time_thickness)
        else:
            keymetrics = time_thickness[['Date','Air_Temp_(C)','Th(snow)','Th(ice)','Th(ocean)']]
            st.dataframe(keymetrics)

        def get_table_download_link(time_thickness):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            csv = time_thickness.to_csv(index=False)
            b64 = base64.b64encode(
            csv.encode()
            ).decode()  # some strings <-> bytes conversions necessary here
            name= "str(self.buoyid) + '_MOSAiC_dataset.csv'"
            if self.buoyid=="T56":
                return f'<a href="data:file/csv;base64,{b64}" download= "T56_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T58":
                return f'<a href="data:file/csv;base64,{b64}" download= "T58_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T62":
                return f'<a href="data:file/csv;base64,{b64}" download= "T62_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T63":
                return f'<a href="data:file/csv;base64,{b64}" download= "T63_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T64":
                return f'<a href="data:file/csv;base64,{b64}" download= "T64_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T65":
                return f'<a href="data:file/csv;base64,{b64}" download= "T65_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T66":
                return f'<a href="data:file/csv;base64,{b64}" download= "T66_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T68":
                return f'<a href="data:file/csv;base64,{b64}" download= "T68_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T70":
                return f'<a href="data:file/csv;base64,{b64}" download= "T70_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T72":
                return f'<a href="data:file/csv;base64,{b64}" download= "T72_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T73":
                return f'<a href="data:file/csv;base64,{b64}" download= "T73_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T74":
                return f'<a href="data:file/csv;base64,{b64}" download= "T74_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T75":
                return f'<a href="data:file/csv;base64,{b64}" download= "T75_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T76":
                return f'<a href="data:file/csv;base64,{b64}" download= "T76_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T77":
                return f'<a href="data:file/csv;base64,{b64}" download= "T77_MOSAiC_dataset.csv">Download csv file</a>'
            elif self.buoyid=="T79":
                return f'<a href="data:file/csv;base64,{b64}" download= "T79_MOSAiC_dataset.csv">Download csv file</a>'

        st.markdown(get_table_download_link(time_thickness), unsafe_allow_html=True)

        
        
        

        
                 
        
      

        #infotext= str(self.buoyid + /n + self.temp + /n + self.heat + /n + self.loc + /n + self.startdate)

        st.header("Data used in analysis")
        if st.button('Links to the MOSAiC Portal'):
            
            st.markdown(self.temp)
            st.markdown(self.heat)
            st.markdown(self.loc)
            st.markdown("Hard-coded start date: " + self.startdate)

        
        


        lat=time_thickness['ts_latitude (deg)'].values.tolist()
        lon=time_thickness['ts_longitude (deg)'].values.tolist()
        thick=time_thickness['Ice_Thickness_(m)'].values.tolist()
        time=time_thickness['ts_time'].values.tolist()
        
        df=pd.DataFrame(zip(lat,lon,thick,time),columns=['lat','lon','Ice Thickness (m)','time'])
        #df=df.tail(1)

        st.header("See the Data on a Map")
        
        #if st.button('Display location'):

#            viewstate=pdk.ViewState(latitude=lat[-1],longitude=lon[-1],zoom=2,pitch=0,)
#            #st.table(df)

#            layer= pdk.Layer('ScatterplotLayer',df,get_position=['lon', 'lat'],get_color='[200, 30, 0, 160]',get_radius=5000)
            
#            st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',initial_view_state=viewstate,layers=[layer]))
        
        import plotly.express as px

        #fig = px.line_mapbox(df, lat="lat", lon="lon", zoom=1, height=300, projection="orthographic")

        fig = px.scatter_geo(df, lat="lat", lon="lon", color='Ice Thickness (m)', 
                  color_continuous_scale=px.colors.cyclical.IceFire, projection="orthographic",width=1000, height=600)

        #fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=1, mapbox_center_lat = 83,autosize=False,width=1000, height=600,margin=dict(l=40, r=40, b=40, t=40))

        st.plotly_chart(fig)
        
        


        
T56=Buoy("T56",
         "https://data.meereisportal.de/download/buoys/2019T56_300234065176750_TEMP_proc.csv",
          "https://data.meereisportal.de/download/buoys/2019T56_300234065176750_HEAT120_proc.csv",
          "https://data.meereisportal.de/download/buoys/2019T56_300234065176750_TS.csv",
         "2019-12-01", "https://data.meereisportal.de/maps/buoys/T/2019T56_image.png"
        )
    
T58=Buoy("T58",
         "https://data.meereisportal.de/download/buoys/2019T58_300234065171790_TEMP_proc.csv",
        "https://data.meereisportal.de/download/buoys/2019T58_300234065171790_HEAT120_proc.csv",
        "https://data.meereisportal.de/download/buoys/2019T58_300234065171790_TS.csv",
         "2019-12-01", "https://data.meereisportal.de/maps/buoys/T/2019T58_image.png"
        )

T62=Buoy("T62",
         "https://data.meereisportal.de/download/buoys/2019T62_300234068706290_TEMP_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T62_300234068706290_HEAT120_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T62_300234068706290_TS.csv",
         "2019-12-01", "https://data.meereisportal.de/maps/buoys/T/2019T62_image.png"
        )

T63=Buoy("T63",
         "https://data.meereisportal.de/download/buoys/2019T63_300234068709320_TEMP_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T63_300234068709320_HEAT120_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T63_300234068709320_TS.csv",
         "2019-12-01", "https://data.meereisportal.de/maps/buoys/T/2019T63_image.png"
        )

T64=Buoy("T64",
        "https://data.meereisportal.de/download/buoys/2019T64_300234068701300_TEMP_proc.csv",
        "https://data.meereisportal.de/download/buoys/2019T64_300234068701300_HEAT120_proc.csv",
        "https://data.meereisportal.de/download/buoys/2019T64_300234068701300_TS.csv",
        "2019-12-01", "https://data.meereisportal.de/maps/buoys/T/2019T64_image.png"
        )

T65=Buoy("T65",
         "https://data.meereisportal.de/download/buoys/2019T65_300234068705730_TEMP_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T65_300234068705730_HEAT120_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T65_300234068705730_TS.csv",
         "2019-12-01" , "https://data.meereisportal.de/maps/buoys/T/2019T65_image.png"
        )

T66=Buoy("T66",
         "https://data.meereisportal.de/download/buoys/2019T66_300234068706330_TEMP_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T66_300234068706330_HEAT120_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T66_300234068706330_TS.csv", 
         "2019-12-01", "https://data.meereisportal.de/maps/buoys/T/2019T66_image.png" 
        )

T68=Buoy("T68",
         "https://data.meereisportal.de/download/buoys/2019T68_300234068708330_TEMP_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T68_300234068708330_HEAT120_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T68_300234068708330_TS.csv",
         "2019-12-01", "https://data.meereisportal.de/maps/buoys/T/2019T68_image.png" 
        )

T70=Buoy("T70",
         "https://data.meereisportal.de/download/buoys/2019T70_300234068705280_TEMP_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T70_300234068705280_HEAT120_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T70_300234068705280_TS.csv", 
         "2019-12-01", "https://data.meereisportal.de/maps/buoys/T/2019T70_image.png" 
        )

T72=Buoy("T72",
         "https://data.meereisportal.de/download/buoys/2019T72_300234068700290_TEMP_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T72_300234068700290_HEAT120_proc.csv",
         "https://data.meereisportal.de/download/buoys/2019T72_300234068700290_TS.csv",
         "2019-12-01", "https://data.meereisportal.de/maps/buoys/T/2019T72_image.png" 
        )

T73=Buoy("T73","https://data.meereisportal.de/download/buoys/2020T73_300234068328160_TEMP_proc.csv",
               "https://data.meereisportal.de/download/buoys/2020T73_300234068328160_HEAT120_proc.csv",
        
            "https://data.meereisportal.de/download/buoys/2020T73_300234068328160_TS.csv",
         "2020-04-15", "https://data.meereisportal.de/maps/buoys/T/2020T73_image.png"
        )

T74=Buoy("T74", "https://data.meereisportal.de/download/buoys/2020T74_300234068225520_TEMP_proc.csv",
       "https://data.meereisportal.de/download/buoys/2020T74_300234068225520_HEAT120_proc.csv",
        
            "https://data.meereisportal.de/download/buoys/2020T74_300234068225520_TS.csv",
         "2020-04-15", "https://data.meereisportal.de/maps/buoys/T/2020T74_image.png"
        )

T75=Buoy("T75","https://data.meereisportal.de/download/buoys/2020T75_300234068325170_TEMP_proc.csv",
       "https://data.meereisportal.de/download/buoys/2020T75_300234068325170_HEAT120_proc.csv",
       
            "https://data.meereisportal.de/download/buoys/2020T75_300234068325170_TS.csv",
         "2020-05-10", "https://data.meereisportal.de/maps/buoys/T/2020T75_image.png"
        )

T76=Buoy("T76","https://data.meereisportal.de/download/buoys/2020T76_300234068528490_TEMP_proc.csv",
       "https://data.meereisportal.de/download/buoys/2020T76_300234068528490_HEAT120_proc.csv",
        
            "https://data.meereisportal.de/download/buoys/2020T76_300234068528490_TS.csv",
         "2020-04-15","https://data.meereisportal.de/maps/buoys/T/2020T76_image.png"
        )

T77=Buoy("T77","https://data.meereisportal.de/download/buoys/2020T77_300234068524740_TEMP_proc.csv",
       "https://data.meereisportal.de/download/buoys/2020T77_300234068524740_HEAT120_proc.csv",
        
            "https://data.meereisportal.de/download/buoys/2020T77_300234068524740_TS.csv",
         "2020-04-15","https://data.meereisportal.de/maps/buoys/T/2020T77_image.png"
        )

T79=Buoy("T79","https://data.meereisportal.de/download/buoys/2020T79_300234068527600_TEMP_proc.csv",
       "https://data.meereisportal.de/download/buoys/2020T79_300234068527600_HEAT120_proc.csv",
        
            "https://data.meereisportal.de/download/buoys/2020T79_300234068527600_TS.csv",
         "2020-04-28", "https://data.meereisportal.de/maps/buoys/T/2020T79_image.png"
        )       

st.title("MOSAiC Buoy Analysis")

st.markdown("This project uses a combination of MOSAiC buoys to generate these figures and datasets, the team and science behind MOSAiC can be found here: https://www.meereisportal.de/en/seaicemonitoring/buoy-mapsdata/") 
st.markdown("The amateur scientists behind this website can be found at https://forum.arctic-sea-ice.net/")
st.markdown("For information on our algorithm see the left panel")
st.markdown("Loading times getting you down? This webapp is hosted on a free AWS-EC2 linux server")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")




st.sidebar.title("Choose a Buoy")
buoyid = st.sidebar.radio("Select Buoy",('T56', 'T58','T62','T63','T64','T65','T66','T68','T70','T72','T73 (new!)','T74 (new!)','T75 (new!)','T76 (new!)','T77 (new!)','T79 (new!)'))

st.sidebar.title('Read the manual')

if st.sidebar.button('Open manual'):

    st.sidebar.title("How it works")
    st.sidebar.markdown("MOSAiC project deploys buoys in the icepack. The buoys consist of a column of thermistors (Th), which are 2cm apart. One set of measurements takes the temperature at each of these thermistors. Another measurement is the thermal capacity surrounding these thermistors. Our algorithm calculates the following values:")
    st.sidebar.header("Th(ice)")
    st.sidebar.markdown("The 'Th_ice' is the thermistor at the boundary between ice and air (or snow) which we calculate by calculating the between-thermistor temperature deviations to determine the point at which ice (low deviation from ocean), transitions to air or snow (high deviation)")
    st.sidebar.header("Th(snow)")
    st.sidebar.markdown("Having calculated 'Th_ice' we define 'Th_snow' as a column of comparatively low temperature standard-deviation between adjacent measurements temporally, as snow insulates thermistors from more variable air temperatures")
    st.sidebar.header("Th(ocean)")
    st.sidebar.markdown("In the seasons of fall, winter and spring, we make the assumption that the ice:ocean interface occurs where the temperature at the thermistor is <-2.2degC, as ocean temperatures under the buoys do not go below -2degC")
    st.sidebar.header("Calculating thickness")
    st.sidebar.markdown("Thickness is calculated by subtracting the Th integers and multiplying by 2cm")
    st.sidebar.header("Limitations")
    st.sidebar.markdown("1) This method will underestimate thickness at the start of a dataset, so we code in start-dates a few weeks post-deployment")
    st.sidebar.markdown("2) Th(snow) requires 150 adjacent measurements to be accurate, so is likely to under-represent day to day variability should it occur")





if buoyid == 'T56':
    buoy = T56
    
elif buoyid == 'T58':
    buoy = T58

elif buoyid == 'T62':
    buoy = T62
    
elif buoyid == 'T63':
    buoy = T63 
    
elif buoyid == 'T64':
    buoy = T64 
    
elif buoyid == 'T65':
    buoy = T65 
    
elif buoyid == 'T66':
    buoy = T66
    
elif buoyid == 'T68':
    buoy = T68
    
elif buoyid == 'T70':
    buoy = T70
    
elif buoyid == 'T72':
    buoy = T72 
 
elif buoyid == 'T73 (new!)':
    buoy = T73 
    
elif buoyid == 'T74 (new!)':
    buoy = T74
    
elif buoyid == 'T75 (new!)':
    buoy = T75
    
elif buoyid == 'T76 (new!)':
    buoy = T76
    
elif buoyid == 'T77 (new!)':
    buoy = T77
    
elif buoyid == 'T79 (new!)':
    buoy = T79

_max_width_()    

buoy.processbuoy()

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("Contributors: SimonF92, uniquorn, oren, blumenkraft, kassy")
st.markdown("Contact email: simonf14@hotmail.co.uk")

