from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
from nilmtk import DataSet, MeterGroup
from nilmtk.utils import print_dict
import pandas as pd





def generate_all_1day_15min_list(h5file):

    dataport = DataSet(h5file)
    num_building = len(dataport.buildings)

    all_main_lists = [] 

    for building_id in range(1, num_building+1):
        elec = dataport.buildings[building_id].elec
        mains = elec.mains()
        mains_timeframe = mains.get_timeframe()
        mains_start_time = mains_timeframe.start
        mains_end_time = mains_timeframe.end

        # Specify the start and end dates
        start_date = mains_start_time.date()
        end_date = mains_end_time.date()
        # Generate the date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')


        main_list = []

        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            new_main = elec.mains().power_series_all_data()[f'{date_str} 00:00:00-05:00':f'{date_str} 23:45:00-05:00'].resample('15T').pad()
            main_list.append(new_main)

                
        all_main_lists.append(main_list)

    return all_main_lists






    
          

    




        


    


    

    






               
           
           


               
        
        
               
 
        

        
        
        
    
    










    
    
    

    
    
    
    