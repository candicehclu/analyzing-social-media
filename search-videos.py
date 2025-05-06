from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pandas as pd
import time

# Replace with your actual API key
API_KEY = 'AIzaSyDoVhxVu1JCl4b6QYPzwflKsH81lZuqBPY'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# Calculate the date six months from today
six_months_ago = datetime.now() - timedelta(days=180)
published_after = six_months_ago.isoformat("T") + "Z"  # Format as ISO 8601

# Build the YouTube service object
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

# Perform the search
def search_youtube(keyword, channel_id, published_after):
    results = []
    next_page_token = None
    while True:
        request = youtube.search().list(
            q=keyword,
            part='snippet',
            channelId=channel_id,
            type='video',
            order='date',
            publishedAfter=published_after,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        results.extend(response.get('items', []))
        next_page_token = response.get('nextPageToken')
        time.sleep(1)
        if not next_page_token:
            break
    return results

# for each news outlet search both 
channel_id = ['UCupvZG-5ko_eiXAupbDfxWw', # CNN (left)
              'UCBi2mrWuNuyYy4gbM6fU18Q', # ABC (left)
              'UC8p1vwvWtl6T73JiExfWs1g', # CBS (left)
              # 'UCP6HGa63sBC7-KHtkme-p-g', # USA today (left)
              # 'UCMliswJ7oukCeW35GSayhRA', # WSJ (neutral)
              # 'UCHjm6wybRbldhqveS7c2WTA', # Insider (left)
              'UCXIJgqnII2ZOINSWNOGFThA', # Fox (right)
              'UCYI_ychRnL7sJrG6PUSBpQA', # CBN (right)
              'UCrvhNP_lWuPIP6QZzJmM-bw', # New York Post (right)
]
channels = ['CNN', 'ABC', 'CBS', 'Fox', 'CBN', 'NYP']
filtered = []

for i in range(len(channel_id)):

    # call api on both keywords
    harvard_results = search_youtube('harvard', channel_id[i], published_after)
    dei_results = search_youtube('dei', channel_id[i], published_after)
    combined_results = harvard_results + dei_results
    seen_ids = set()

    # append to data
    for item in combined_results:
        title = item['snippet']['title']
        video_id = item['id']['videoId']
        published_at = item['snippet']['publishedAt']

        if video_id in seen_ids:
            continue

        # if not seen, add video
        seen_ids.add(video_id)

        filtered.append({
            'title': title,
            'video_id': video_id,
            'published_at': published_at,
            'channel_title': channels[i]
        })

df = pd.DataFrame(filtered)
df.to_csv('harvard_dei_6.csv', index=False)