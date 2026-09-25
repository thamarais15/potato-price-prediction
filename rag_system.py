import pandas as pd
import numpy as np

class PotatoPriceRAG:
    def __init__(self, csv_path):
        try:
            self.df = pd.read_csv(csv_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.df = pd.DataFrame()

    def retrieve_context(self, query):
        if self.df.empty:
            return "I'm sorry, I don't have access to the price data right now."
            
        query = query.lower()
        
        # 1. GLOBAL STATS (High-level insights)
        if any(word in query for word in ["all", "summary", "overall", "general", "overview"]):
            summary = (
                f"Overall Market Summary:\n"
                f"- Price Range: {self.df['price'].min():.2f} to {self.df['price'].max():.2f} units\n"
                f"- Average Price: {self.df['price'].mean():.2f} units\n"
                f"- Most Volatile Period: {self.df['price'].std():.2f} (Std Dev)\n"
                f"- Total Days Analyzed: {len(self.df)} days"
            )
            return summary

        # 2. EXTREMES (Highest/Lowest)
        if any(word in query for word in ["highest", "maximum", "max", "peak"]):
            max_row = self.df.loc[self.df['price'].idxmax()]
            return (f"Peak Price Alert: The highest recorded price was {max_row['price']:.2f} units "
                    f"on {max_row['date'].date()}. At that time, temperature was {max_row['temperature']:.1f}°C "
                    f"and rainfall was {max_row['rainfall']:.1f}mm.")

        if any(word in query for word in ["lowest", "minimum", "min", "bottom"]):
            min_row = self.df.loc[self.df['price'].idxmin()]
            return (f"Price Floor: The lowest recorded price was {min_row['price']:.2f} units "
                    f"on {min_row['date'].date()}. At that time, temperature was {min_row['temperature']:.1f}°C "
                    f"and rainfall was {min_row['rainfall']:.1f}mm.")

        # 3. AVERAGES
        if any(word in query for word in ["average", "mean", "typical"]):
            avg_price = self.df['price'].mean()
            return f"The typical market price for potatoes is approximately {avg_price:.2f} units."

        # 4. TIME-BASED FILTERS (Months/Years)
        # Check for month names
        months = ['january', 'february', 'march', 'april', 'may', 'june', 
                  'july', 'august', 'september', 'october', 'november', 'december']
        
        found_month = next((m for m in months if m in query), None)
        if found_month:
            month_num = months.index(found_month) + 1
            month_data = self.df[df['date'].dt.month == month_num]
            if not month_data.empty:
                return (f"Insights for {found_month.capitalize()}:\n"
                        f"- Avg Price: {month_data['price'].mean():.2f} units\n"
                        f"- Max Price: {month_data['price'].max():.2f} units\n"
                        f"- Min Price: {month_data['price'].min():.2f} units\n"
                        f"- Sample Date: On {month_data.iloc[0]['date'].date()}, price was {month_data.iloc[0]['price']:.2f}")

        # 5. SPECIFIC DATE SEARCH
        # Search for YYYY-MM-DD patterns
        import re
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
        if date_match:
            target_date = date_match.group(1)
            row = self.df[self.df['date'].dt.strftime('%Y-%m-%d') == target_date]
            if not row.empty:
                r = row.iloc[0]
                return (f"Data for {target_date}:\n"
                        f"- Price: {r['price']:.2f} units\n"
                        f"- Temp: {r['temperature']:.1f}°C\n"
                        f"- Rainfall: {r['rainfall']:.1f}mm\n"
                        f"- Demand Index: {r['demand_index']:.2f}")

        return "I couldn't find a specific match. You can ask me about 'highest prices', 'average prices', 'summary of all data', or prices for a specific month (e.g., 'January')."

    def generate_answer(self, query):
        context = self.retrieve_context(query)
        
        # Simulating an LLM that synthesizes the context
        if "summary" in query.lower() or "all" in query.lower():
            return f"Here is the complete market overview:\n\n{context}"
        
        if "highest" in query.lower() or "lowest" in query.lower():
            return f"Based on the historical peak/bottom analysis:\n\n{context}"
        
        return f"Analyzing the records for your query...\n\n{context}"
