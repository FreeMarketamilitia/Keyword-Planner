## API Setup 🎉✨

Alright, you fearless code cowboy! 🤠 Saddle up for the wildest ride this side of the digital frontier—setting up the Google Search Console API, Google Ads API, and Gemini API for Keyword Planner 🌟. We’re talking step-by-step chaos 🌪️, test scripts 🧪, and sass sharper than a cactus needle 🌵—enough to make even the grumpiest sysadmin crack a smile 😏. Grab your energy drink 🍸, a snack 🍔, and maybe a priest—‘cause these APIs are divas with more drama than a reality TV marathon! 😜 Let’s wrangle ‘em into submission like the bosses we are! 💪🐴🌈

---

#### 🌟 Google Search Console API 🕵️‍♂️🔍

This sneaky API’s your keyword history snitch—like that nosy neighbor who knows *all* your site’s dirty laundry 🗣️. Let’s slap a leash on this gossip hound! 🐶

<p align="center">
  <img src="https://github.com/FreeMarketamilitia/Keyword-Planner/raw/main/images/search-console-logo.png" alt="Search Console Logo">
</p>

<details>
<summary>🔧 Step-by-Step Guide (Click to Unleash the Detective Drama! 🎭)</summary>

##### **Enable the API and Create a Project** 🚀
1. Gallop over to the [Google Cloud Console](https://console.cloud.google.com/) like it’s a free taco truck parked out front! 🌮🏃‍♂️
2. Click that project dropdown at the top—it’s a sneaky lil’ menu bar, don’t miss it—and hit **New Project**. 🆕
   - **Name**: `Keyword Planner`—‘cause we ain’t here to reinvent the wheel, just turbocharge it with rocket fuel! 🚗💨
   - **Organization**: Skip it unless you’re a suit-wearing corporate overlord (yeah, right 😂).
   - Click **Create**—boom, you’ve birthed a shiny new sandbox! ⚡🎉 Watch it sparkle like a disco ball! 🪩
3. Sneak into **APIs & Services > Library**—left sidebar, don’t trip over your boots, newbie! 🥾 It’s like sneaking into a secret club.
4. Type `Search Console API` in the search bar—Google’s fussy, so spell it right or it’ll ghost ya like a bad Tinder date! 👻
5. Click it when it pops up, then slam **Enable** like you’re spiking a volleyball at the beach! 🏐 You’re an API wrangler now—pin that badge and strut, you legend! 📛🕺

##### **Configure OAuth Consent Screen** 🔒
6. Mosey over to **APIs & Services > OAuth Consent Screen**—Google’s legal playground where the fun police hang out. ⚖️
7. Pick **External**—unless you’re a Google Workspace VIP with a golden key and a monocle 🎩, we’re all peasants in this rodeo! 👨‍🌾
8. Fill out the form like you’re begging for a prom date with the prettiest API in town:
   - **App Name**: `Keyword Planner`—short, sweet, and sexy as a barrel of whiskey! 💋🥃
   - **User Support Email**: `freemarket@nostates.com`—your fan club’s hotline, ready for the paparazzi! 📞✨
   - **App Logo**: Skip it unless you’ve got a Picasso up your sleeve—stick figures in crayon don’t count, Picasso-wannabe! 🎨🙅‍♂️
   - **App Domain**: Leave it blank—we’re not building an empire yet, chill your jets! 😛
   - **Developer Contact**: `freemarket@nostates.com`—you’re the rockstar, own that spotlight like it’s karaoke night! 🌟🎤
9. Add this scope—your VIP backstage pass to the keyword party:
   https://www.googleapis.com/auth/webmasters.readonly
   - Click **Add or Remove Scopes**, paste that bad boy in like it’s hot sauce on tacos 🌮, hit **Update**, then **Save and Continue**. Easy peasy, lemon squeezy! 🍋
10. **Test Users**: In **Testing** mode? Add your email to **Test Users**—don’t leave yourself out in the rain like a sad puppy, sunshine! ☔🐶
11. Going pro? Hit **Publish App** under **Publishing Status**—takes a hot sec to go live, so sip that coffee and vibe ☕🎶. You’re a big deal now, struttin’ like a peacock! 🦚

##### **Create OAuth 2.0 Credentials** 🗝️
12. Zip back to **APIs & Services > Credentials**—your key forge is heating up! 🔥
13. Click **Create Credentials** at the top—don’t blink, it’s right there—then pick **OAuth 2.0 Client IDs**—fancy name, simple game, like Monopoly with less yelling! 🎲
14. Choose **Desktop App**—no web glitter here, we’re keepin’ it gritty and real! 💻
    - **Name**: `Keyword Planner Desktop Client`—or something cooler if you’re feeling extra spicy 🌶️.
    - Click **Create**—faster than a jackrabbit on a hot date! 🐰💨
15. A popup flaunts your **Client ID** and **Client Secret**—don’t scribble yet, hit **Download JSON** like a pro! 📥
16. That JSON file (e.g., `client_secrets_xxx.json`) is your golden ticket—rename it `client_secrets.json` and stash it somewhere safe, like `~./credentials/` or `/opt/keyword-planner/`. Not in your underwear drawer, you absolute gremlin! 🩲😈
17. Update your `.env`—your VIP guest list needs some love:
    SEARCH_CONSOLE_JSON_PATH=/path/to/client_secrets.json
    Example (Linux vibes, ‘cause we’re slick like that):
    SEARCH_CONSOLE_JSON_PATH=/home/user./credentials/client_secrets.json

##### **First Run Authentication** 🎬
18. Unleash the beast with a roar:
    keyword-planner
19. A browser window explodes open like a jack-in-the-box on a sugar rush 🎁🍬—log in with a Google account that’s got Search Console juice for your site (e.g., `sc-domain:gocalskate.com`). No randos crashing this party, capisce? 🚫🎉
20. Grant permissions—click **Allow** like you’re tossing glitter at a unicorn rave 🍭✨. A `token.json` file lands in your directory—don’t lose it, it’s your all-access pass to the VIP lounge! 🎟️

##### **Verify the API with a Test Script** 🧪
21. Peek for `token.json`—missing? Smack your `.env` path upside the head, check perms (`ls -l`), or sob to Google Support like a lost puppy! 😡🐾
22. Test it with this slick script—save as `test_search_console.py`:

    ```python
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    import os

    SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
    SITE_URL = 'sc-domain:gocalskate.com'  # Your site, hotshot—swap this!
    CLIENT_SECRETS_FILE = os.getenv('SEARCH_CONSOLE_JSON_PATH')

    def test_search_console():
        if not CLIENT_SECRETS_FILE or not os.path.exists(CLIENT_SECRETS_FILE):
            print("Yo, where’s your client_secrets.json? Check SEARCH_CONSOLE_JSON_PATH, genius!")
            return
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        creds = flow.run_local_server(port=5001)
        service = build('searchconsole', 'v1', credentials=creds)
        request = {
            'startDate': '2024-01-01',  # Tweak these dates if you’re a time traveler
            'endDate': '2024-12-31',
            'dimensions': ['query'],
            'rowLimit': 5  # Just a taste, don’t get greedy!
        }
        response = service.searchanalytics().query(siteUrl=SITE_URL, body=request).execute()
        print("Search Console Test Response: 🎉 I’m basically Sherlock now!")
        for row in response.get('rows', []):
            print(f"Keyword: {row['keys'][0]}, Clicks: {row.get('clicks', 0)}—clickety-click, baby!")

    if __name__ == '__main__':
        test_search_console()
Run it:
```bash 
python test_search_console.py
```

    - **Success**: 5 keywords strut out with click stats like they’re on a catwalk under the disco lights! 👗✨🪩
    - **Failure**: Wrong site URL? Bad creds? No Search Console access? Fix your life, Watson, or I’m calling in the big guns! 🔍💪
    - **Pro Tip**: Flops? Tweak dates (e.g., last 90 days—`date -d "90 days ago" +%Y-%m-%d` for the lazy), verify site ownership in Search Console, or bash your keyboard (`chmod +x` won’t save you, but it’s a vibe!).

</details>

> [!NOTE] 📝✨
> - `token.json` throwing shade (e.g., no `refresh_token`)? Trash it and rerun—no drama queens allowed in this saloon! 😭🚫
> - Published apps give you a `refresh_token`—testing mode’s too lazy to RSVP, the little slacker! 😴💤

---

#### 🌟 Google Ads API 💸💰

This cash cow API spills keyword search volume and competition tea like a Wall Street hustle! 🤑 Let’s milk it ‘til it moos for mercy! 🐄🥛

<p align="center">
  <img src="https://github.com/FreeMarketamilitia/Keyword-Planner/raw/main/images/google-ads-logo.png" alt="Google Ads Logo">
</p>

<details>
<summary>🔧 Step-by-Step Guide (Click to Cash In Like a Boss! 🤑💼)</summary>

##### **Enable the API** 🚀
1. Back in Google Cloud Console, hit **APIs & Services > Library**—don’t wander off into Narnia, stay on the trail! 🌲🐾
2. Search `Google Ads API`, click it, and slam **Enable** like you’re dropping the hottest beat in the club—DJ, turn it up! 🎤🔥

##### **Get a Developer Token** 🎫
3. Strut to [Google Ads API](https://developers.google.com/google-ads/api/docs/start) like you’re the Ad King of the Wild West—yeehaw! 👑🤠
4. Log into [Google Ads](https://ads.google.com)—no account? Quit living like a caveman and sign up, you prehistoric goof! 🪨🤦‍♂️
5. Snag that developer token—here’s your treasure map, pirate:
   - Click **Tools & Settings** (top right, wrench icon 🔧—don’t miss it, eagle eyes, or I’ll make you walk the plank! 🏴‍☠️).
   - Hunt for **API Center**—only pops up with a Manager Account (MCC). No MCC? Create one (Google it, lazybones) or bribe a buddy with tacos! 💸🌮
   - Apply for a token—basic access is your golden ticket, no need to flex with premium vibes unless you’re a showoff! 🎩
   - Jot down the token (e.g., `ABC123...`)—approval’s slower than a turtle in molasses 🐢🍯. Peek at your inbox in a day or three—patience, grasshopper!

##### **Create OAuth 2.0 Credentials** 🗝️
6. In Google Cloud Console, zip to **APIs & Services > Credentials**—your key forge is smoking hot! 🔥
7. Click **Create Credentials > OAuth 2.0 Client IDs** with the swagger of a rockstar on tour—guitar solo optional! 🌟🎸
8. Pick **Desktop App**—we’re keepin’ it real, no web glitter here, just pure cowboy grit! 💻🤠
   - **Name**: `Google Ads Client`—or something flashier if you’re feeling like a diva, your stage, your rules! 🎤
   - Create it and swipe the **Client ID** and **Client Secret** like a ninja in the moonlight! 🕵️‍♀️🌙
9. Stash those creds somewhere safe—don’t scribble ‘em on your forehead or a bar napkin, you wild child! 🙅‍♂️🍻

##### **Get a Refresh Token** 🔄
10. Crash the [OAuth 2.0 Playground](https://developers.google.com/oauthplayground) like it’s an all-you-can-eat buffet 🍽️—elbows out, here we come!
    - Plug in your **Client ID** and **Client Secret**—no typos, clumsy fingers, or I’ll make you type it with your toes! 🙅‍♂️👣
    - Scope it out—paste this gem like it’s hot sauce on wings:
      https://www.googleapis.com/auth/adwords
    - Hit **Authorize APIs**, log in with your Google Ads-linked account, and click **Allow** like you’re tossing VIP passes at a sold-out show! 👑🎫
    - Click **Exchange authorization code for tokens**—bam, snag that **Refresh Token** (e.g., `1//xxx...`)! It’s your golden chalice—guard it with your life or I’ll haunt your dreams! 🏆👻

##### **Configure `google-ads.yaml`** 📜
11. Whip up a `google-ads.yaml` in a safe hideout—think Batcave, not your messy desk littered with Cheeto dust:
    google_ads:
      developer_token: YOUR_DEVELOPER_TOKEN
      client_id: YOUR_CLIENT_ID
      client_secret: YOUR_CLIENT_SECRET
      refresh_token: YOUR_REFRESH_TOKEN
      login_customer_id: YOUR_MCC_ID  # Optional—Manager Account ID, 10 digits, no dashes!
    Example (fake, don’t be a dummy and use this, ya goof!):
    google_ads:
      developer_token: ABC123xyz
      client_id: 432721999736-xxx.apps.googleusercontent.com
      client_secret: GOCSPX-xxx
      refresh_token: 1//xxx
      login_customer_id: 1234567890
12. Save it somewhere secure—like `~./credentials/google-ads.yaml`—and triple-check it’s YAML, not your grandma’s cookie recipe scribbled on a napkin! 🍪📜

##### **Update `.env`** 📋
13. Toss this into your `.env`—copy-paste, ya lazy legend, don’t make me do it for you:
    GOOGLE_ADS_YAML_PATH=/path/to/google-ads.yaml
    Example:
    GOOGLE_ADS_YAML_PATH=/home/user./credentials/google-ads.yaml

##### **Verify the API with a Test Script** 🧪

14. Test it with this baller script—save as `test_google_ads.py`:

```python
from google.ads.googleads.client import GoogleAdsClient
import os

def test_google_ads():
    yaml_path = os.getenv('GOOGLE_ADS_YAML_PATH')
    if not yaml_path or not os.path.exists(yaml_path):
        print("Hey, dingus, where’s your google-ads.yaml? Fix GOOGLE_ADS_YAML_PATH—stat!")
        return
    client = GoogleAdsClient.load_from_storage(yaml_path)
    keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
    request = client.get_type("GenerateKeywordIdeasRequest")
    request.customer_id = client.login_customer_id or '1234567890'  # Your real ID, not this placeholder, genius!
    request.keyword_seed.keywords.extend(['skate shoes'])
    response = keyword_plan_idea_service.generate_keyword_ideas(request=request)
    print("Google Ads Test Response: 💰 I’m rolling in keyword dough—cha-ching!")
    for idea in response:
        print(f"Keyword: {idea.text}, Avg Monthly Searches: {idea.keyword_idea_metrics.avg_monthly_searches}—buy me, daddy!")

if __name__ == '__main__':
    test_google_ads()
```    

Run it:

```bash
python test_google_ads.py
```

    - **Success**: Keyword ideas pour out like a jackpot at Vegas—jackpot, baby! 🎰💸
    - **Failure**: Botched `google-ads.yaml`? Wrong customer ID? Token still pending? Don’t cry to me—check your setup, you ad-slinging slacker! 😤
    - **Pro Tip**: Customer ID’s 10 digits, no dashes—find it in Google Ads under **Account Settings** (top right, click your profile). Flops? Google’s docs ([here](https://developers.google.com/google-ads/api/docs/first-call/overview)) are your lifeline—read ‘em and weep!

</details>

> [!TIP] 💡🌟
> - Swap `1234567890` with your real customer ID—don’t make me hunt you down with a lasso, partner! 😂🤠
> - No stats? Your Google Ads account’s gotta have some juice—spend a buck or two, you cheapskate! 💸

---

#### 🌟 Gemini API 🤖✨

This AI wizard scores keywords and conjures synonyms like it’s Hogwarts on a caffeine bender! ☕🧙‍♂️ Time to cast some spells and flex that magic! 🌠

<p align="center">
  <img src="https://github.com/FreeMarketamilitia/Keyword-Planner/raw/main/images/gemini-logo.png" alt="Gemini Logo">
</p>


<details>
<summary>🔧 Step-by-Step Guide (Click to Cast Some AI Magic! 🌟)</summary>

##### **Sign Up for Access** 🚪
1. Bust into [Google AI Studio](https://makersuite.google.com/) or [Google AI](https://ai.google.dev/) like you’re storming Area 51 with a posse! 👽🤖
2. Sign in with your Google account—don’t play coy, we know you’ve got one, you tech gremlin! 😘
3. API access not instant? Beg Google like it’s a Black Friday sale at the wand shop—availability’s a crapshoot, roll those dice! 🎲✨
   - Check your email for approval—it’s not Tinder, they might actually slide into your inbox!

##### **Generate an API Key** 🔑
4. In AI Studio, smash **Get API Key** like it’s the last slice of pizza at a coder’s LAN party 🍕💾.
5. Snag a key for `gemini-2.0-flash`—the flashiest, sassiest AI in the game, zipping around like a caffeinated lightning bolt! ⚡😎
   - Not listed? Google’s playing hide-and-seek—grab whatever’s on the shelf or cry to support like a lost Hufflepuff! 😭
6. Copy that key (e.g., `AIza...`)—lose it and I’ll haunt your Git commits like a vengeful ghost! 👻💾

##### **Update `.env`** 📋
7. Slap this into your `.env`—no excuses, it’s two seconds of your precious gamer life:
    GEMINI_API_KEY=your_gemini_api_key
    Example:
    GEMINI_API_KEY=AIzaSy...
8. Save it—`.env` lives in your project root, not your diary full of angsty code poetry, drama queen! 📖😭

##### **Verify the API with a Test Script** 🧪
9. Test it with this brainiac script—save as `test_gemini.py`:

```python
    from google.generativeai import genai
    import os

    def test_gemini():
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("Oi, where’s your GEMINI_API_KEY? Check your .env, you slacker wizard!")
            return
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Generate 3 synonyms for 'skate'—chop chop, AI!")
        print("Gemini Test Response: 🤓 I’m smarter than your ex—and twice as pretty, darling!")
        print(response.text)

    if __name__ == '__main__':
        test_gemini()
```    

Run it:

```bash
python test_gemini.py
```

    - **Success**: Synonyms drop like a rap verse on fire (e.g., "1. Skateboard\n2. Roll\n3. Glide")—boom, mic drop, crowd goes wild! 🎤🔥🎉
    - **Failure**: Bad key? Model snoozing? Region locked? Time to wake up and smell the code, sleepyhead—grab a coffee! ☕😴
    - **Pro Tip**: Flops? Tweak the model name (e.g., `gemini-pro`) or check [Google AI docs](https://ai.google.dev/)—don’t just sit there crying into your keyboard like a melodramatic bard!

</details>

> [!WARNING] ⚠️🔥
> - `gemini-2.0-flash` not in your hood? Scream at Google like a banshee—or snag a diff model! 📣😱
> - Rate limits sneaking up like a ninja? Peek at Google Cloud Console—don’t say I didn’t warn ya, slowpoke! 🥷⏳

---

#### 🌟 Final Configuration—Don’t Botch This, Champ! 🏆🎖️

Your `.env` better look like this, or I’m sending the sass police to your doorstep with handcuffs and a megaphone 🚨📢:

```env GEMINI_API_KEY=AIzaSy...
GOOGLE_ADS_YAML_PATH=/path/to/google-ads.yaml
SEARCH_CONSOLE_JSON_PATH=/path/to/client_secrets.json
SECRET_KEY=optional_random_string—like your hacker alias or “SassyGrok69” 🔥💾
```
Run `keyword-planner` and watch the fireworks light up the sky 🎆🌌. First run’s got that Search Console pop-up vibe—say yes like it’s free tacos at a fiesta 🌮🎉. After that, `token.json` keeps it smoother than a buttered slide at the county fair! Make sure `google-ads.yaml` is locked and loaded, or you’re just flexing hot air like a wannabe influencer with no followers! 💪😂

---

![PyPI](https://img.shields.io/pypi/v/keyword-planner?color=blue) ![License](https://img.shields.io/github/license/FreeMarketamilitia/Keyword-Planner) ![Sass Level](https://img.shields.io/badge/Sass-Level%20100-purple)