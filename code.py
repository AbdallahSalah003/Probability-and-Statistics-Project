import opendatasets as od
dataset_url = 'https://www.kaggle.com/datasets/saife245/english-premier-league/data'
od.download(dataset_url)
data_file = '/content/english-premier-league/2021-2022.csv'
import pandas as pd
df = pd.read_csv(data_file)
misssing_data = df.isna().sum().sort_values(ascending=False) / len(df) * 100
misssing_data[misssing_data!=0] 
# shows that there is no missing values in columns that we will use
# Showing the Percentages of missing values in columns
import matplotlib.pyplot as plt
misssing_data[misssing_data!=0].plot(kind='bar')
plt.title('Columns Contains missing Values')
plt.xlabel('Columns')
plt.ylabel('Percentage')
df1 = pd.read_csv(data_file)
# Define the queries
queries = [
    'Receive yellow or red cards',
    'Has Awarded fouls',
    'Has Awarded Corners',
    'Have Shots',
    'Have Shots on target'
]
# Initialize an empty dictionary to store frequencies
frequency_dict = {query: 0 for query in queries}
# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Determine the winner based on goal scores
    if row['FTHG'] > row['FTAG']:
        winner = 'Away'
    elif row['FTHG'] < row['FTAG']:
        winner = 'Home'
    else:
        winner = 'Draw'
    
    # Check if the condition is met for each query and update the frequency
    for query in queries:
        if query in ['Receive yellow or red cards', 'Has Awarded fouls']:
            if (winner == 'Home' and (row['HY'] > 0 or row['HR'] > 0)) or (winner == 'Away' and (row['AY'] > 0 or row['AR'] > 0)):
                frequency_dict[query] += 1
        elif query == 'Has Awarded Corners':
            if (winner == 'Home' and row['HC'] > 0) or (winner == 'Away' and row['AC'] > 0):
                frequency_dict[query] += 1
        elif query == 'Have Shots':
            if (winner == 'Home' and row['HS'] > 0) or (winner == 'Away' and row['AS'] > 0):
                frequency_dict[query] += 1
        elif query == 'Have Shots on target':
            if (winner == 'Home' and row['HST'] > 0) or (winner == 'Away' and row['AST'] > 0):
                frequency_dict[query] += 1

# Convert the dictionary to a DataFrame
frequency_df = pd.DataFrame(list(frequency_dict.items()), columns=['Winner', 'Frequency'])

#Winner and Loser Calculations
import numpy as np
def summary_stats(df):
    # Create an empty DataFrame to store the results
    result = pd.DataFrame()
    # Loop through each column in the input DataFrame
    for col in df.columns:
        # Calculate the mean, median, mode, range, standard deviation, and variance for each column
        mean = round(df[col].mean(),2)
        median = round(df[col].median(),2)
        mode = round(df[col].mode()[0],2) # Take the first mode if there are multiple modes
        range = round(df[col].max() - df[col].min(),2)
        std =round( df[col].std(),2)
        var = round(df[col].var(),2)
        # Append the results to the result DataFrame
        result[col] = [mean, median, mode, range, std, var]
    # Add row labels
    result.index = ['Mean', 'Median', 'Mode', 'Range', 'Std', 'Var']
    # Return the result DataFrame
    return result
data = {
    '#Goals': [], 
    'Shots': [],
    'Shots on T': [],
    'Fouals': [],
    'Corners': [],
    'Y Cards': [],
    'R Cards': [],
}

# Create a dataframe from the dictionary
df3 = pd.DataFrame(data)
for ind in df.index:
    if(df['FTR'][ind] == 'A'): 
        df3.loc[len(df3)] = [df['FTHG'][ind],df['HS'][ind],df['HST'][ind],df['AF'][ind],df['HC'][ind],df['HY'][ind],df['HR'][ind]]
    elif(df['FTR'][ind] == 'H'):
        df3.loc[len(df3)] = [df['FTAG'][ind],df['AS'][ind],df['AST'][ind],df['HF'][ind],df['AC'][ind],df['AY'][ind],df['AR'][ind]]

summary_stats(df3)

dataW = {
    'ST': [],
    'WOL': [],
}
df4= pd.DataFrame(data2)
for ind in df.index:
    if(df['FTR'][ind] == 'A'): 
        df4.loc[len(df4)] = [df['AST'][ind],'W']
        df4.loc[len(df4)] = [df['HST'][ind],'L']
    elif(df['FTR'][ind] == 'H'):
        df4.loc[len(df4)] = [df['AST'][ind],'L']
        df4.loc[len(df4)] = [df['HST'][ind],'W']

# Group the DataFrame by the Result column and sum the Shots on Target column
df_grouped = df4.groupby('WOL')['ST'].sum()

# Plot a pie chart with labels, colors, percentages, and explode effect
df_grouped.plot(kind='pie', labels=['Loser', 'Winner'], colors=['red', 'green'], autopct='%1.1f%%', explode=[0.1, 0])

# Set the title and the aspect ratio of the pie chart
plt.title('Awarded Fouals of Winner and Loser Teams')
plt.axis('equal')

# Show the pie chart
plt.show()

#plotting Bundeslegs VS EPL
BundWST = []
BundLST = []
EPLWST = []
EPLLST = []
for ind in df2.index:
    if(df['FTR'][ind] == 'A'): 
        BundWST.append(df2['AST'][ind])
        BundLST.append(df2['HST'][ind])
    elif(df['FTR'][ind] == 'H'):
        BundWST.append(df2['HST'][ind])
        BundLST.append(df2['AST'][ind])
for ind in df.index:
    if(df['FTR'][ind] == 'A'): 
        EPLWST.append(df['AST'][ind])
        EPLLST.append(df['HST'][ind])
    elif(df['FTR'][ind] == 'H'):
        EPLWST.append(df['HST'][ind])
        EPLLST.append(df['AST'][ind])
plt.boxplot([BundWST,EPLWST],vert=False)
plt.title('EPL Winner Shots on Target VS Bundeslega Winner Shots on Target')

BundWRC = []
BundLRC = []
EPLWRC = []
EPLLRC = []
for ind in df2.index:
    if(df['FTR'][ind] == 'A'): 
        BundWRC.append(df2['A'][ind])
        BundLRC.append(df2['HR'][ind])
    elif(df['FTR'][ind] == 'H'):
        BundWRC.append(df2['HR'][ind])
        BundLRC.append(df2['AR'][ind])
for ind in df.index:
    if(df['FTR'][ind] == 'A'): 
        EPLWRC.append(df['AR'][ind])
        EPLLRC.append(df['HR'][ind])
    elif(df['FTR'][ind] == 'H'):
        EPLWRC.append(df['HR'][ind])
        EPLLRC.append(df['AR'][ind])
plt.boxplot([BundWRC,EPLWRC],vert=False)
plt.title('EPL Winner Red Cards VS Bundeslega Winner Red Cards')

#possession on histograms
Wposs = []
Lposs = []
for ind in df9.index:
    if(df9['gf'][ind]>df9['ga'][ind]): 
        Wp.append(df9['poss'][ind])
    elif(df9['gf'][ind]<df9['ga'][ind]):
        Lp.append(df9['poss'][ind]) 
# Generate a histogram with 5 bins
plt.hist(Lp, bins=10, density=True,alpha=0.5,color='r',label='Loser')
plt.hist(Wp, bins=10, density=True,label='Winner')

# Add labels and title
plt.xlabel('Penalty kicks attempted')
plt.ylabel('Frequency')
plt.title('Loser and Winner Penalty kicks attempted in EPL')


#compare betwwen possesstion % and winning 
possession = [[] for i in range(100)]
for ind in df9.index:
    if(df9['result'][ind]=='W'): possession[df9['poss'][ind].astype(int)].append([df9['gf'][ind]-df9['ga'][ind]])

possession2 = [[] for i in range(100)]
for ind in range(len(possession)):
    sum = 0
    for ind2 in range(len(possession[ind])):
        sum=sum+possession[ind][ind2][0];
    if(len(possession[ind])>0): 
        possession2[ind]=round(sum/len(possession[ind]),2)

posss = []
margins = []
for ind in range(len(possession2)):
    if possession2[ind]:
        posss.append(ind)
        margins.append(possession2[ind])
corr_coef = np.corrcoef(margins, posss)


#red cards vs winning 
rc = []
margins2 = []
for ind in df.index:
    if(df['FTR'][ind]=='H'): 
        rc.append(df['HR'][ind])
        margins2.append(df['FTHG'][ind]-df['FTAG'][ind])
    elif( df['FTR'][ind]=='A' ):
        rc.append(df['AR'][ind])
        margins2.append(df['FTAG'][ind]-df['FTHG'][ind])
import matplotlib.pyplot as plt
plt.scatter(rc,margins2)
plt.xlabel('Red Cards')
plt.ylabel('Margin of Vectory')
plt.title('Winning VS #Red Cards')
plt.show()
from scipy.stats import pearsonr
ans = pearsonr(rc, margins2)
print("Pearson correlation coefficient:", ans)

