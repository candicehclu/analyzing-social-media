from googleapiclient.discovery import build
import pandas as pd
import time

# Replace this with YOUR API key (guard it like your online dignity)
api_key = 'AIzaSyDoVhxVu1JCl4b6QYPzwflKsH81lZuqBPY'

# Build the YouTube client
youtube = build('youtube', 'v3', developerKey=api_key)

def get_all_comments(video_id):
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText',
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment_snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'comment': comment_snippet['textDisplay'],
                'likes': comment_snippet['likeCount'],
                'published_at': comment_snippet['publishedAt'],
                'replies': comment_snippet['totalReplyCount']
            })

        # Check if there’s another page
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

        # Optional: be polite to the API (not strictly needed, but good practice)
        time.sleep(0.1)

    return comments

# Example usage:
video_id = 'wjHtvAYKCto'
comments_data = get_all_comments(video_id)

# cbs: uyT4GLl5lDI
# dw: gZSxDDqVZO0
# fox: kj8cneOkLdM
# cnn: wjHtvAYKCto

# Save to CSV
df = pd.DataFrame(comments_data)
df.to_csv('cnn_harvard.csv', index=False)
print(f"Saved")
