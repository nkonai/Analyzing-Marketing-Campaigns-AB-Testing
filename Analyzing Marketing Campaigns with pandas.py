#!/usr/bin/env python
# coding: utf-8

# # Analyzing Marketing Campaigns with pandas

# In[ ]:


# Import pandas into the environment
import pandas as pd

# Import marketing.csv 
marketing = pd.read_csv('marketing.csv')


# ## Examining the data

# In[ ]:


# Print the first five rows of the DataFrame
print(marketing.head())


# In[ ]:


# Print the statistics of all columns
print(marketing.describe())


# In[ ]:


# Check column data types and non-missing values
print(marketing.info())


# ## Updating the data type of a column

# In[ ]:


# Check the data type of is_retained
print(marketing['is_retained'].dtype)


# In[ ]:


# Convert is_retained to a boolean
marketing['is_retained'] = marketing['is_retained'].astype('bool')

# Check the data type of is_retained, again
print(marketing['is_retained'].dtype)


# ## Adding New Columns

# In[ ]:


# Mapping for channels
channel_dict = {"House Ads": 1, "Instagram": 2, 
                "Facebook": 3, "Email": 4, "Push": 5}

# Map the channel to a channel code
marketing['channel_code'] = marketing['subscribing_channel'].map(channel_dict)


# In[ ]:


# Import numpy
import numpy as np

# Add the new column is_correct_lang
marketing['is_correct_lang'] = np.where(marketing['language_displayed']==marketing['language_preferred'],'Yes','No')

marketing['is_correct_lang'].head()


# ## Date columns

# In[ ]:


# Import pandas into the environment
import pandas as pd

# Import marketing.csv with date columns
marketing = pd.read_csv('marketing.csv',parse_dates=['date_served','date_subscribed','date_canceled'])

# Add a DoW column
marketing['DoW'] = marketing['date_subscribed'].dt.dayofweek


# ## Daily marketing reach by channel

# In[ ]:


# Group by date_served and count number of unique user_id's
daily_users = marketing.groupby(['date_served'])['user_id'].nunique()

# Print head of daily_users
print(daily_users.head())


# In[ ]:


# Plot daily_subscribers
daily_users.plot()

# Include a title and y-axis label
plt.title('Daily users')
plt.ylabel('Number of users')
plt.xlabel('Date')

# Rotate the x-axis labels by 45 degrees
plt.xticks(rotation = 45)

# Display the plot
plt.show()


# ## Calculating conversion rate

# In[ ]:


# Calculate the number of people we marketed to
total = marketing['user_id'].nunique()

# Calculate the number of people who subscribed
subscribers = marketing[marketing['converted']==True]['user_id'].nunique()

# Calculate the conversion rate
conversion_rate = subscribers/total
print(round(conversion_rate*100, 2), "%")


# ## Calculating retention rate

# In[ ]:


# Calculate the number of subscribers
total_subscribers = marketing[marketing['converted']==True]['user_id'].nunique()
print(total_subscribers)

# Calculate the number of people who remained subscribed
retained = marketing[marketing['is_retained']==True]['user_id'].nunique()
print(retained)

# Calculate the retention rate
retention_rate = retained/total_subscribers
print(round(retention_rate*100, 2), "%")


# ## Comparing language conversion rate (I)

# In[ ]:


# Isolate english speakers
english_speakers = marketing[marketing['language_displayed'] == 'English']

# Calculate the total number of English speaking users
total = english_speakers['user_id'].nunique()
print(total)

# Calculate the number of English speakers who converted
subscribers = english_speakers[marketing['converted']==True]['user_id'].nunique()
print(subscribers)

# Calculate conversion rate
conversion_rate = subscribers/total
print('English speaker conversion rate:', round(conversion_rate*100,2), '%')


# In[ ]:


# Group by language_displayed and count unique users
total = marketing.groupby(['language_displayed'])['user_id'].nunique()
print(total)

# Group by language_displayed and count unique conversions
subscribers = marketing[marketing['converted']==True].groupby('language_displayed')['user_id'].nunique()
print(subscribers)

# Calculate the conversion rate for all languages
language_conversion_rate = subscribers/total
print(round(language_conversion_rate*100,2),'%')


# ## Aggregating by date

# In[ ]:


# Group by date_served and count unique users
total = marketing.groupby(['date_served'])['user_id'].nunique()
print(total)
# Group by date_served and count unique converted users
subscribers = marketing[marketing['converted']==True].groupby(['date_served'])['user_id'].nunique()
print(subscribers)

# Calculate the conversion rate per day
daily_conversion_rate = subscribers/total
print(round(daily_conversion_rate*100,2),'%')


# ## Visualize conversion rate by language

# In[ ]:


# Create a bar chart using language_conversion_rate DataFrame
language_conversion_rate.plot(kind='bar')

# Add a title and x and y-axis labels
plt.title('Conversion rate by language\n', size = 16)
plt.xlabel('Language', size = 14)
plt.ylabel('Conversion rate (%)', size = 14)

# Display the plot
plt.show()


# ## Creating daily conversion rate DataFrame

# In[ ]:


# Group by date_served and count unique users
total = marketing.groupby(['date_served'])['user_id']                     .nunique()

# Group by date_served and calculate subscribers
subscribers = marketing[marketing['converted'] == True]                         .groupby(['date_served'])                         ['user_id'].nunique()

# Calculate the conversion rate for all languages
daily_conversion_rates = subscribers/total


# ## Visualize daily conversion rate

# In[ ]:


# Create a line chart using daily_conversion_rate
print(daily_conversion_rate)
daily_conversion_rate.plot('date_subscribed','conversion_rate')

plt.title('Daily conversion rate\n', size = 16)
plt.ylabel('Conversion rate (%)', size = 14)
plt.xlabel('Date', size = 14)

# Set the y-axis to begin at 0
plt.ylim(0)

# Display the plot
plt.show()


# ## Marketing channels across age groups

# In[ ]:


channel_age = marketing.groupby(['marketing_channel', 'age_group'])                                ['user_id'].count()
print(channel_age)

# Unstack channel_age and transform it into a DataFrame
channel_age_df = pd.DataFrame(channel_age.unstack(level = 1))
print(channel_age_df)
# Plot channel_age
channel_age_df.plot(kind = 'bar')
plt.title('Marketing channels by age group')
plt.xlabel('Age Group')
plt.ylabel('Users')
# Add a legend to the plot
plt.legend(loc = 'upper right', 
           labels = channel_age_df.columns.values)
plt.show()


# ## Grouping and counting by multiple columns

# In[ ]:


# Count the subs by subscribing channel and day
retention_total = marketing.groupby(['date_subscribed',
                                     'subscribing_channel'])['user_id'].nunique()


# Print results
print(retention_total.head())


# In[ ]:


# Sum the retained subs by subscribing channel and date subscribed
print(marketing)
retention_subs = marketing[marketing['is_retained']==True].groupby(['date_subscribed', 
                                       'subscribing_channel'])['user_id'].nunique()

# Print results
print(retention_subs.head())


# In[ ]:


# Divide retained subscribers by total subscribers
retention_rate = retention_subs/retention_total
print(retention_rate)
retention_rate_df = pd.DataFrame(retention_rate.unstack(level=1))
print(retention_rate_df)

# Plot retention rate
retention_rate_df.plot()

# Add a title, x-label, y-label, legend and display the plot
plt.title('Retention Rate by Subscribing Channel')
plt.xlabel('Date Subscribed')
plt.ylabel('Retention Rate (%)')
plt.legend(loc = 'upper right',labels = retention_rate_df.columns.values)
plt.show()


# ## Building a conversion function

# In[ ]:


def conversion_rate(dataframe, column_names):
    # Total number of converted users
    column_conv = dataframe[dataframe['converted']==True].groupby(column_names)['user_id'].nunique()

    # Total number users
    column_total = dataframe.groupby(column_names)['user_id'].nunique()
    
    # Conversion rate 
    conversion_rate = column_conv/column_total
    
    # Fill missing values with 0
    conversion_rate = conversion_rate.fillna(0)
    return conversion_rate


# ## Test and visualize conversion function

# In[ ]:


# Calculate conversion rate by age_group
age_group_conv = conversion_rate(marketing, ['date_served', 'age_group'])
print(age_group_conv)

# Unstack and create a DataFrame
age_group_df = pd.DataFrame(age_group_conv.unstack(level=1))
print(age_group_df)
# Visualize conversion by age_group
age_group_df.plot()
plt.title('Conversion rate by age group\n', size = 16)
plt.ylabel('Conversion rate', size = 14)
plt.xlabel('Age group', size = 14)
plt.show()


# ## Plotting function

# In[ ]:


def plotting_conv(dataframe):
    for column in dataframe:
        # Plot column by dataframe's index
        plt.plot(dataframe.index,dataframe[column])
        plt.title('Daily ' + str(column) + ' conversion rate\n', 
                  size = 16)
        plt.ylabel('Conversion rate', size = 14)
        plt.xlabel('Date', size = 14)
        # Show plot
        plt.show()  
        plt.clf()


# ## Putting it all together

# In[ ]:


# Calculate conversion rate by date served and age group
age_group_conv = conversion_rate(marketing,['date_served','age_group'])

# Unstack age_group_conv and create a DataFrame
age_group_df = pd.DataFrame(age_group_conv.unstack(level=1))

# Plot the results
plotting_conv(age_group_df)


# ## House ads conversion rate

# In[ ]:


# Calculate conversion rate by date served and channel
daily_conv_channel = conversion_rate(marketing,['date_served','marketing_channel'])

print(daily_conv_channel.head())


# In[ ]:


# Calculate conversion rate by date served and channel
daily_conv_channel = conversion_rate(marketing, ['date_served', 
                                                 'marketing_channel'])

# Unstack daily_conv_channel and convert it to a DataFrame
daily_conv_channel = pd.DataFrame(daily_conv_channel.unstack(level = 1))

# Plot results of daily_conv_channel
plotting_conv(daily_conv_channel)


# ## Analyzing House ads conversion rate

# In[ ]:


# Add day of week column to marketing
marketing['DoW_served'] = marketing['date_served'].dt.dayofweek


# Calculate conversion rate by day of week
DoW_conversion = conversion_rate(marketing, ['DoW_served', 'marketing_channel'])
print(DoW_conversion)

# Unstack channels
DoW_df = pd.DataFrame(DoW_conversion.unstack(level=1))

# Plot conversion rate by day of week
DoW_df.plot()
plt.title('Conversion rate by day of week\n')
plt.ylim(0)
plt.show()


# ## House ads conversion by language

# In[ ]:


# Isolate the rows where marketing channel is House Ads
house_ads = pd.DataFrame(marketing[marketing['marketing_channel']=='House Ads'])
print(house_ads)

# Calculate conversion by date served, and language displayed
conv_lang_channel = conversion_rate(house_ads,['date_served','language_displayed'])
print(conv_lang_channel)

# Unstack conv_lang_channel
conv_lang_df = pd.DataFrame(conv_lang_channel.unstack(level=1))
print(conv_lang_df)

# Use your plotting function to display results
plotting_conv(conv_lang_df)


# ## Creating a DataFrame for house ads

# In[ ]:


# Add the new column is_correct_lang
house_ads['is_correct_lang'] = np.where(
    house_ads['language_displayed'] == house_ads['language_preferred'], 
    'Yes', 
    'No')
print(house_ads)

# Groupby date_served and correct_language
language_check = house_ads.groupby(['date_served','is_correct_lang'])['user_id'].count()

print(language_check)

# Unstack language_check and fill missing values with 0's
language_check_df = pd.DataFrame(language_check.unstack(level=1)).fillna(0)

# Print results
print(language_check_df)


# ## Confirming house ads error

# In[ ]:


# Divide the count where language is correct by the row sum
language_check_df['pct'] = language_check_df['Yes']/language_check_df.sum(axis=1)


# Plot and show your results
plt.plot(language_check_df.index.values, language_check_df['pct'])
plt.show()


# ## Setting up conversion indexes

# In[ ]:


# Calculate pre-error conversion rate
house_ads_bug = house_ads[house_ads['date_served'] < '2018-01-11']
lang_conv = conversion_rate(house_ads_bug,['language_displayed'])


# Index other language conversion rate against English
spanish_index = lang_conv['Spanish']/lang_conv['English']
arabic_index = lang_conv['Arabic']/lang_conv['English']
german_index = lang_conv['German']/lang_conv['English']

print("Spanish index:", spanish_index)
print("Arabic index:", arabic_index)
print("German index:", german_index)


# ## Analyzing user preferences

# In[ ]:


# Group house_ads by date and language
converted = house_ads.groupby(['date_served','language_preferred'])                        .agg({'user_id':'nunique',
                              'converted':'sum'})

# Unstack converted
converted_df = pd.DataFrame(converted.unstack(level=1))
converted_df


# ## Creating a DataFrame based on indexes

# In[ ]:


# Create English conversion rate column for affected period
converted['english_conv_rate'] = converted.loc['2018-01-11':'2018-01-31'][('converted','English')]

# Create expected conversion rates for each language
converted['expected_spanish_rate'] = converted['english_conv_rate'] * spanish_index
converted['expected_arabic_rate'] = converted['english_conv_rate'] * arabic_index
converted['expected_german_rate'] = converted['english_conv_rate'] * german_index


# Multiply number of users by the expected conversion rate
converted['expected_spanish_conv'] = converted['expected_spanish_rate']*converted[('user_id','Spanish')]/100
converted['expected_arabic_conv'] = converted['expected_arabic_rate']*converted[('user_id','Arabic')]/100
converted['expected_german_conv'] = converted['expected_german_rate']*converted[('user_id','German')]/100


# ## Assessing bug impact

# In[ ]:


# Use .loc to slice only the relevant dates
converted = converted.loc['2018-01-11':'2018-01-31']

# Sum expected subscribers for each language
expected_subs = converted['expected_spanish_conv'].sum() + converted['expected_arabic_conv'].sum() + converted['expected_german_conv'].sum()

# Calculate how many subscribers we actually got
actual_subs = converted[('converted','Spanish')].sum() + converted[('converted','Arabic')].sum() + converted[('converted','German')].sum()

# Subtract how many subscribers we got despite the bug
lost_subs = expected_subs - actual_subs
print(lost_subs)


# ## A/B Testing - Test allocation

# In[ ]:


# Subset the DataFrame
email = marketing[marketing['marketing_channel']=='Email']
email

# Group the email DataFrame by variant 
alloc = email.groupby('variant')['user_id'].nunique()
alloc
# Plot a bar chart of the test allocation
alloc.plot(kind='bar')
plt.title('Personalization test allocation')
plt.ylabel('# participants')
plt.show()


# ## Comparing conversion rates

# In[ ]:


# Group marketing by user_id and variant
subscribers = email.groupby(['user_id', 
                             'variant'])['converted'].max()
subscribers_df = pd.DataFrame(subscribers.unstack(level=1)) 

# Drop missing values from the control column
control = subscribers_df['control'].dropna()

# Drop missing values from the personalization column
personalization = subscribers_df['personalization'].dropna()

print('Control conversion rate:', np.mean(control))
print('Personalization conversion rate:', np.mean(personalization))


# ## Creating a lift function

# In[ ]:


def lift(a,b):
    # Calcuate the mean of a and b
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    
    # Calculate the lift using a_mean and b_mean
    lift = (b_mean - a_mean)/a_mean
  
    return str(round(lift*100, 2)) + '%'
  
# Print lift() with control and personalization as inputs
print(lift(control, personalization))


# ## Evaluating statistical significance

# In[ ]:


stats.ttest_ind(control,personalization)


# ## Building an A/B test segmenting function

# In[ ]:


def ab_segmentation(segment):
  # Build a for loop for each segment in marketing
  for subsegment in np.unique(marketing[segment].values):
      print(subsegment)
      
      # Limit marketing to email and subsegment      
      email = marketing[(marketing['marketing_channel'] == 'Email') & (marketing[segment] == subsegment)]

      subscribers = email.groupby(['user_id', 'variant'])['converted'].max()
      subscribers = pd.DataFrame(subscribers.unstack(level=1)) 
      control = subscribers['control'].dropna()
      personalization = subscribers['personalization'].dropna()

      print('lift:', lift(control, personalization))
      print('t-statistic:', stats.ttest_ind(control, personalization), '\n\n')


# ## Using your segmentation function

# In[ ]:


# Use ab_segmentation on language displayed
ab_segmentation('language_displayed')


# In[ ]:


# Use ab_segmentation on age group
ab_segmentation('age_group')

