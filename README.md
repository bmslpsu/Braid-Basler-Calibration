## This fcuntion is to do automatic single point calibration for Braid system with high accuracy.
May extend this function to replace easywand for Photron cameras in the future.

## Data collection:
### wand points:
1. Run braid without any calibration xml files and set detecting **<u>1</u>** point max for all the cameras.
2. Open terminal, `braid run /path/to/toml`.
3. Look for the Braid URL to open a browser and click `save .braidz`.
4. Wave the 1 led wand through the whole volume like a tornado, the finer, the more accurate.
5. Click the button again to stop saving.
6. Save/rename the saved `.braidz` file as you like

### real scale reference points
1. Run braid without any calibration xml files and set detecting **<u>2</u>** points max for all the cameras.
2. Open terminal, `braid run /path/to/toml`.
3. Look for the Braid URL to open a browser and open the view for all the cameras.
4. **important**: set the 2-led stand in camera views and make sure the cameras are tracking the 2 points correctly.   
It is okay to have at least 2 cameras to have a clear view and correct tracking of the two leds.  
**check the views and make sure the tracking is correct, this is extreme important for the accuracy of autoscale**
5. Click `save .braidz` and wait for a couple of seconds (do not move anything).
6. Click the button again to stop saving.
7. Save/rename the saved `.braidz` file as you like

## How to use the code:
1. use `unzip_braidz.py` to unzip your braidz file.
2. use `multicam_calibration.py` to do calibration.
3. use `auto_adjust_scale` with reference points to adjust the scale to real.
4. use `modify_real_coor_gui.py` to adjust the coordinate from the last step to real coordinate.
5. may use `test_xml_reprojection_error.py` to verify the xml accuracy.
