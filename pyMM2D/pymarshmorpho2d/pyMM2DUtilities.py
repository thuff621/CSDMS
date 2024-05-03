# This script supports the main script with file handling, formatting and other tasks.
def modelGridProjection(crs, fldir):
    from glob import glob
    import rasterio as rio
    import numpy as np
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(fill_value=np.nan, strategy='mean')
    fls = glob(''.join([fldir, '/*.asc']))
    print(f'{len(fls)} model files found to be projected')
    for pathhr in fls:
        outputName = ''.join([str(pathhr).split('.')[0], '_projected.tif'])
        with rio.open(pathhr) as src:
            profile = src.profile.copy()
            transform, width, height = calculate_default_transform(
                src.crs, crs, src.width, src.height, *src.bounds)
            profile.update({
                'crs': crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rio.open(outputName, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    data = np.nan_to_num(src.read(i))
                    # Or
                    data = imputer.fit_transform(data)  # use array not pandas df

                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.nearest
                    )
def calculateRasterStats(grid, fldir, classNames, rasterClassNumbers):
    from glob import glob
    import numpy as np
    import rasterio as rio
    import pandas as pd

    print("Calculating raster statistics")
    fls = glob(''.join([fldir, '/*.tif']))
    fl = []
    for f in fls:
        if str(f).find('land_cover_change_projected.tif') != -1:
            fl.append(f)
    # find the last file in the list
    for i in range(0, len(fl)):
        for f in fl:
            if str(f).find(''.join([str(i), '_land_cover_change_projected.tif'])) != -1:
                lstFile = f
    print(f'{lstFile} detected to be the last file in the sequence')
    with rio.open(lstFile) as ds:
        ar = ds.read()
        height = ds.height
        width = ds.width
    # extract data based on the code
    # land cover types 0,1,2,3 (water, low marsh, transitional, high marsh)
    # there is a 1 added to the front of the code string.  IGNORE THE FIRST 1 in the first position.
    # The code works by taking the second number as the land cover class at the beginning of the model and the final number
    # is the land cover class at the end of the model.  So 132 is starting at high marsh (3) and going to transitional (2).  Again ignore the first 1.
    classes = classNames # ['water', 'healthy_marsh', 'moderate_marsh', 'struggling_marsh', 'transitional', 'high_marsh']
    numericalClasses = rasterClassNumbers # [0, 1, 2, 3, 4, 5]
    # build a confusion matrix to allow for start and end array comparisons
    df = np.zeros(shape=(len(numericalClasses), len(numericalClasses)))
    df[:] = np.nan
    for stClass in numericalClasses:
        for endClass in numericalClasses:
            df[endClass, stClass] = len(
                ar[np.where(ar == int(''.join([str(1), str(stClass), str(endClass)])), True, False)])
    stCols = []
    for cc in classes:
        stCols.append(''.join([cc, '_start_class']))
    end = []
    for cc in classes:
        end.append(''.join([cc, '_end_class']))
    pdDf = pd.DataFrame(df, columns=stCols)
    pdDf.set_index([end], inplace=True)
    pdDf.to_csv(''.join([fldir, '/LandCover_Change_Stats.csv']))
    # print the stats
    area = height / grid.shape[0] * width / grid.shape[1]
    for n in numericalClasses:
        all = sum(pdDf.iloc[:, n])
        orig = pdDf.iloc[n, n]
        changeArea = (all - orig) * area
        if changeArea > 0:
            print(f'There was a {(changeArea)} meter change for class {classes[n]}')
            print(f'With {orig} meters of {classes[n]} recorded')
        else:
            print(f"no change detected for class {classes[n]}")

def saveModelProgression(mev, dir, rnd):
    import os
    from landlab.io.esri_ascii import write_esri_ascii
    nm = str(dir).split('/')[-1]
    flnm = ''.join([nm,'_', str(rnd), '.asc'])
    fldir = ''.join([dir, '/modelProgressionFiles'])
    if os.path.exists(fldir) == False:
        os.makedirs(fldir, exist_ok=True)
        print(f'Creating model progression directory {fldir}')

    files = write_esri_ascii(''.join([fldir, '/', flnm]), mev.grid)
    [os.path.basename(name) for name in sorted(files)]

def plotModelRuns(dir, rasterClassNumbers, classNames, limits, saveFile):
    from glob import glob
    import numpy as np
    import rasterio as rio
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm

    fls = glob(''.join([dir, '/*_land_cover_projected.tif']))
    print(f'A total of {len(fls)} were found.')
    for i in range(0, len(fls)):
        fl = glob(''.join([dir, '/*_', str(i), '_land_cover_projected.tif']))
        try:
            with rio.open(fl[0]) as ds:
                ar = ds.read()
                height = ds.height
                width = ds.width
            lst = []
            for cl in rasterClassNumbers:
                lst.append(len(ar[np.where(ar == cl, True, False)]))
        except:
            print("File not found")
        if i == 0:
            hd = pd.DataFrame(zip(rasterClassNumbers, lst), columns = ['numerical_class', ''.join(['timeStep_', str(i), '_sq_meters'])])
        else:
            hd[''.join(['timeStep_', str(i), '_sq_meters'])] = lst
    # convert to percent change
    ptchange = hd.copy()
    for i in range(1, len(hd.columns)):
        ptchange.iloc[:,i] = ((ptchange.iloc[:,i] / hd.iloc[:,1]) * 100) - 100
    # plot each class
    pltLen =  len(ptchange.iloc[1,:])
    color = ['blue', 'green', 'tan', 'brown', 'red']#iter(cm.rainbow(np.linspace(0,1, len(ptchange.iloc[:,0]))))
    plt.figure(figsize=(8,6))
    for cl in range(0, len(rasterClassNumbers)):
        c = color[cl]
        plt.plot(list(range(1, pltLen)), ptchange.iloc[cl,1:], color = c)
        plt.ylim(-limits,limits)
    plt.legend(classNames, loc = 'upper left')
    plt.xlabel('model time in years')
    plt.ylabel('percent change')
    plt.savefig(saveFile, dpi=300)
    plt.show()

def extractAsCSV(dir, saveFile, observationPoints = None, gridData = None):
    from glob import glob
    import numpy as np
    import rasterio as rio
    import pandas as pd
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from matplotlib.pyplot import cm

    if gridData == None: # this allows you to define the data that you actually want to extract from the data previously saved out
        fls = glob(''.join([dir, '/*.tif'])) # extract ALL data
    else:
        fls = glob(''.join([dir, '/*_', gridData, '.tif'])) # extract a specific dataset
    print(f'A total of {len(fls)} were found.')
    if observationPoints != None:
        for i in range(0, len(fls)):
            fl = glob(''.join([dir, '/*_', str(i), '_', gridData, '.tif']))
            src = rio.open(fl[0])
            if i == 0:
                pts = gpd.read_file(observationPoints)
                # check that the pts crs and the raster crs are the same before proceeding
                if src.crs == pts.crs:
                    print("Projections match. Moving to point extraction")
                else:
                    print(f"Warning! The CRS was not a match. The raster is {src.crs} and the points are in {pts.crs}")
            coord_list = [(x, y) for x, y in zip(pts["geometry"].x, pts["geometry"].y)]
            # coords = pts.get_coordinates()
            pts[''.join(['timestep_', str(i), '_', gridData])] = np.nan # setup an empty column
            pts[''.join(['timestep_', str(i),'_', gridData])] = np.array([(x) for x in src.sample(coord_list)]).astype(float)
        nm = str(saveFile).replace(".csv", '_observationPointExt.csv')
    else: # give the mean value across the entire tif
        dat = []
        tstep = []
        for i in range(0, len(fls)):
            fl = glob(''.join([dir, '/*_', str(i), '_', gridData, '.tif']))
            src = rio.open(fl[0])
            dat.append(np.mean(src.read()))
            tstep.append(''.join(['timestep_', str(i), '_mean']))
        pts = pd.DataFrame(zip(tstep, dat), columns = ['timestep', 'mean_value'])
        nm = saveFile

    print("Saving out csv extraction file")
    print(nm)
    pts.to_csv(nm, index=None)

def extractLandcover(dir, rasterClassNumbers, classNames):
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    from glob import glob
    import os
    import rasterio as rio
    pd.options.mode.chained_assignment = None
    # Give the file directory for exported enterum files.
    i = 0
    fl = glob(''.join([dir, '/*_', str(0),'_land_cover_projected.tif']))
    zeros  = np.zeros(len(rasterClassNumbers))
    df = pd.DataFrame(zip(classNames, zeros), columns=['landcover_class_names',''.join(['timestep_', str(i)])])
    while len(fl) >  0:
        try:
            print(f"Working on file {''.join([dir, '/*_', str(i),'_land_cover_projected.tif'])}")
            fl = glob(''.join([dir, '/*_', str(i),'_land_cover_projected.tif']))
            src = rio.open(fl[0])
            dat = src.read()
            # create an enmpy column
            df[''.join(['timestep_', str(i)])] = np.nan
            for cc in range(0, len(rasterClassNumbers)):
                df[''.join(['timestep_', str(i)])].iloc[cc] = len(dat[dat == rasterClassNumbers[cc]])
        except:
            print(f"Skipping {''.join([dir, '/*', str(i),'_land_cover_projected.tif'])}")
        i += 1
        fl = glob(''.join([dir, '/*', str(i), '_land_cover_projected.tif']))
    df.to_csv(''.join([dir, '/Landcover_change_over_time.csv']), index=None)


def modelProgressionPlot(startRas, endRas, plotName):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import rasterio
    import numpy as np
    import pandas as pd

    # Load the two raster images
    raster1 = rasterio.open(startRas)
    raster2 = rasterio.open(endRas)

    # load in data
    fileStr = str(startRas).split('/')[0:-1]
    flnm = ''
    for f in fileStr:
        flnm = '/'.join([flnm, f])
    flnm = flnm.replace('/examples', 'examples')
    data = pd.read_csv(''.join([flnm, '/Landcover_change_over_time.csv']))



    ptchange = data.copy()
    for i in range(0, len(data.iloc[:,0])):
        val = data.iloc[i,1]
        if val == 0:
            st = 0
            for cc in data.iloc[i,1:]:
                if cc != 0 and st == 0:
                    val = cc
                    st = 1
        ptchange.iloc[i,1:] = ((ptchange.iloc[i,1:] / val) * 100) - 100

    xList = []
    for i in range(0, len(data.iloc[0,:])):
        xList.append(i)

    # Create a figure with two subplots
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(10)

    ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=1)
    ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 1), colspan=1)
    ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), colspan=2)


    # Display the two raster images in the subplots
    axmap1 = ax1.imshow(raster1.read(1), cmap='terrain')
    axmap2 = ax2.imshow(raster2.read(1), cmap='terrain')

    fig1 = fig.colorbar(axmap1, ticks = range(0, len(data.iloc[:,0])))
    fig2 = fig.colorbar(axmap2, ticks = range(0, len(data.iloc[:,0])))

    fig1.ax.set_yticklabels(data.iloc[:,0])
    fig2.ax.set_yticklabels(data.iloc[:, 0])


    # zero list
    zerolist = []
    for r in range(0, len(xList[1:])):
        zerolist.append(0)

    # Add a plot beneath the two raster images
    ax3.plot(xList[1:], ptchange.iloc[0, 1:].astype(float), color='blue', label = ptchange.iloc[0,0])
    ax3.plot(xList[1:], ptchange.iloc[1, 1:].astype(float), color='seagreen', label = ptchange.iloc[1,0])
    ax3.plot(xList[1:], ptchange.iloc[2, 1:].astype(float), color='lime', label=ptchange.iloc[2, 0])
    ax3.plot(xList[1:], ptchange.iloc[3, 1:].astype(float), color='tan', label=ptchange.iloc[3, 0])
    ax3.plot(xList[1:], zerolist,  '--', color='black')
    # ax3.plot(xList[1:], ptchange.iloc[4, 1:].astype(float), color='brown', label=ptchange.iloc[4, 0])
    ax3.plot(xList[1:], ptchange.iloc[5, 1:].astype(float), color='red', label=ptchange.iloc[5, 0])
    ax3.set_ylabel('percent change')
    ax3.set_xlabel('years')
    ax3.set_ylim(-100,100)

    # ax3.plot(xList[1:], ptchange.iloc[4, 1:].astype(float), color='red', label = ptchange.iloc[4,0])

    ax3.legend()

    # Add a title to the figure
    fig.suptitle(plotName, fontsize=14)

    # Display the figure
    plt.tight_layout()
    plt.savefig(''.join([flnm, '/', plotName, '.jpg']), dpi = 300)
    plt.show()

def imageMaker(dir):
    """
    This takes the save model progression function files and converts them to jpg images and then into videos.
    :return:
    """



