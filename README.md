# traffic-accident-risk-prediction
ML model for predicting traffic accident risk zones to optimize emergency response times


## Field Description

| Column | Description |
| :--- | :--- |
| **ID** | This is a unique identifier of the accident record. |
| **Source** | Source of raw accident data. |
| **Severity** | Shows the severity of the accident (1-4). 1 indicates least impact on traffic; 4 indicates significant impact. |
| **Start_Time** | Shows start time of the accident in local time zone. |
| **End_Time** | Shows end time of the accident in local time zone (when traffic impact was dismissed). |
| **Start_Lat** | Shows latitude in GPS coordinate of the start point. |
| **Start_Lng** | Shows longitude in GPS coordinate of the start point. |
| **End_Lat** | Shows latitude in GPS coordinate of the end point. |
| **End_Lng** | Shows longitude in GPS coordinate of the end point. |
| **Distance(mi)** | The length of the road extent affected by the accident in miles. |
| **Description** | Shows a human provided description of the accident. |
| **Street** | Shows the street name in address field. |
| **City** | Shows the city in address field. |
| **County** | Shows the county in address field. |
| **State** | Shows the state in address field. |
| **Zipcode** | Shows the zipcode in address field. |
| **Country** | Shows the country in address field. |
| **Timezone** | Shows timezone based on the location of the accident (Eastern, Central, etc.). |
| **Airport_Code** | Denotes the closest airport-based weather station to the accident location. |
| **Weather_Timestamp** | Shows the time-stamp of weather observation record (in local time). |
| **Temperature(F)** | Shows the temperature (in Fahrenheit). |
| **Wind_Chill(F)** | Shows the wind chill (in Fahrenheit). |
| **Humidity(%)** | Shows the humidity (in percentage). |
| **Pressure(in)** | Shows the air pressure (in inches). |
| **Visibility(mi)** | Shows visibility (in miles). |
| **Wind_Direction** | Shows wind direction. |
| **Wind_Speed(mph)** | Shows wind speed (in miles per hour). |
| **Precipitation(in)** | Shows precipitation amount in inches, if any. |
| **Weather_Condition** | Shows the weather condition (rain, snow, thunderstorm, fog, etc.). |
| **Amenity** | POI annotation indicating presence of amenity nearby. |
| **Bump** | POI annotation indicating presence of speed bump or hump nearby. |
| **Crossing** | POI annotation indicating presence of crossing nearby. |
| **Give_Way** | POI annotation indicating presence of give_way nearby. |
| **Junction** | POI annotation indicating presence of junction nearby. |
| **No_Exit** | POI annotation indicating presence of no_exit nearby. |
| **Railway** | POI annotation indicating presence of railway nearby. |
| **Roundabout** | POI annotation indicating presence of roundabout nearby. |
| **Station** | POI annotation indicating presence of station nearby. |
| **Stop** | POI annotation indicating presence of stop nearby. |
| **Traffic_Calming** | POI annotation indicating presence of traffic_calming nearby. |
| **Traffic_Signal** | POI annotation indicating presence of traffic_signal nearby. |
| **Turning_Loop** | POI annotation indicating presence of turning_loop nearby. |
| **Sunrise_Sunset** | Shows the period of day (day or night) based on sunrise/sunset. |
| **Civil_Twilight** | Shows the period of day based on civil twilight. |
| **Nautical_Twilight** | Shows the period of day based on nautical twilight. |
| **Astronomical_Twilight** | Shows the period of day based on astronomical twilight. |
