''' Processes initial data from xarray.Dataset onto the appropriate grid'''

import numpy as np
from TensorDynamics.vertical_interpolation import pressure_to_sigma

def preprocess_data(model,data):
	"""
	Given xarray with data, interpolate to gaussian grid and model sigma levels
	Args:
		model: model object
		data: xarray dataset with model variables at coordinates (time, level, latitude, longitude)

	Returns:
		states: list of dictionaries with atmospheric states that can be used to run a forecast
	"""

	# horizontal interpolation
	data=data.interp(latitude=model.f_obj.lats*180/np.pi,longitude=model.f_obj.lons*180/np.pi)
	data=data.transpose("time","level","latitude","longitude")

	# for each unique time in the dataset, perform the vertical interpolation to sigma coordinates
	states=[]
	for i in range(len(data["time"])):
		states.append({})
		for vari in ["u_component_of_wind","v_component_of_wind","temperature","specific_humidity"]:
			states[i][vari]=pressure_to_sigma(data["surface_pressure"][i].values,
											 data["level"].values,model.sigmas,data[vari][i].values)

	# append surface variables to already created dictionaries
	for i in range(len(data["time"])):
		for vari in ["surface_pressure","geopotential_at_surface"]:
			states[i][vari]=data[vari][i].values[None,:,:]

	return states



