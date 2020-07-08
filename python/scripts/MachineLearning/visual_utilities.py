
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def graficar_violin_plot (x,y,df,titulo,y_label):
    
    fig = go.Figure()
    niveles = df[x].unique()
    for nivel in niveles:
        xaxis=df[x][df[x] == nivel]
        yaxis=df[y][df[x] == nivel]
        fig.add_trace(go.Violin(x=xaxis, #keep cereal type at x axis
                            y=yaxis, #keep carbohydrates type at y axis
                            name=nivel, #name of each category
                            box_visible=True, #if you want to show box plot within the violin
                            meanline_visible=True, #if meanline of boxplot should be visible
                            points="all" #plot all the points of distribution
                           ))
    fig.update_layout(title = {'text':titulo, 'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},
                  yaxis_title=y_label, #set y axis
                 ) 
    fig.show()
    
def grafica_dispersion(base,variable_eje_x, variable_eje_y,label_x,label_y,titulo,linea_ver=0,linea_hor=0,variable_eje_z="ninguna"):
    
    x_plot = base[[variable_eje_x,variable_eje_y]].values
    
    if variable_eje_z == "ninguna":
        plt.scatter(x_plot[:,0],x_plot[:,1],s=20, c='red')
    else:
        colores = ['blue','red','green','purple','brown','black']
        z_plot = base[variable_eje_z]
        labels = z_plot.unique()
        for i in range(0,len(labels)):
            plt.scatter(x_plot[z_plot==labels[i],0],x_plot[z_plot==labels[i],1],s=20, c=colores[i])
    plt.title(titulo)
    if variable_eje_z != "ninguna":
        plt.legend(labels)
    
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if linea_hor > 0: 
        plt.axhline(y=linea_hor)
    if linea_ver > 0:
        plt.axvline(x=linea_ver)
    plt.show()
    
def bubble_chart(x,y,var_dummy_para_conteo,df,titulo):
    #Create temporary dataframe
    df_agg = df.groupby([x,y],as_index=False).agg({var_dummy_para_conteo:["count"]})
    df_agg.columns = [x,y,"count"]
    df_agg['proporcion'] = df_agg.groupby([x])['count'].transform(lambda x: x/x.sum())
    df_agg['proporcion_string'] = pd.Series(["{0:.2f}%".format(val * 100) for val in df_agg['proporcion']], index = df_agg.index)
    df_agg['proporcion'] = pd.Series([round(val, 3) for val in df_agg['proporcion']], index = df_agg.index)
    

    
    #Plotting
    fig = px.scatter(df_agg, #dataframe where data is stored
                    x=x, #column of df for x axis
                    y=y, #column for df for y axis
                    size="count", #column of df to set size of bubble
                    color="proporcion",
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    color_continuous_midpoint=0.5,
                    hover_name="proporcion_string", #column of df to show text on hover of bubble
                    size_max=40 #maximum size of the bubble
                    )

    fig.update_layout(title = titulo) #set the title

    #to show the plot
    fig.show()

def bubble_chart_2d(x,y,z,var_dummy_para_conteo,df,titulo):
    #Create temporary dataframe
    df_agg = df.groupby([x,y,z],as_index=False).agg({var_dummy_para_conteo:["count"]})
    df_agg.columns = [x,y,z,"count"]
    df_agg['proporcion'] = df_agg.groupby([z])['count'].transform(lambda x: x/x.sum())
    df_agg['proporcion_string'] = pd.Series(["{0:.2f}%".format(val * 100) for val in df_agg['proporcion']], index = df_agg.index)
    df_agg['proporcion'] = pd.Series([round(val, 3) for val in df_agg['proporcion']], index = df_agg.index)
    

    #Plotting
    fig = px.scatter(df_agg, #dataframe where data is stored
                    x=x, #column of df for x axis
                    y=y, #column for df for y axis
                    size="count", #column of df to set size of bubble
                    color="proporcion",
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    color_continuous_midpoint=0.5,
                    hover_name="proporcion_string", #column of df to show text on hover of bubble
                    facet_col=z,
                    size_max=40 #maximum size of the bubble
                    )

    fig.update_layout(title = titulo) #set the title

    #to show the plot
    fig.show()