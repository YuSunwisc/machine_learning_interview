import asyncio
import time

async def brewCoffee():
    print("Start of brewCoffee()")
    await asyncio.sleep(2)
    print("End of brewCoffee()")
    return "Coffe Ready"
    

async def toastBagel():
    print("Start of toastBagel()")
    await asyncio.sleep(1)
    print("End of toastBagel()")
    return "Baggle Ready"

async def main():
    start_time = time.time()

    # 1. Method 1
    coffee_task = asyncio.create_task(brewCoffee())
    toast_task = asyncio.create_task(toastBagel())
    
    result_coffee = await coffee_task
    result_baggle = await toast_task

    # 2. Method 2
    # batch = asyncio.gather(brewCoffee(), toastBagel())
    # result_coffee, result_baggle = await batch

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Result of brewCoffee: {result_coffee}")
    print(f"Result of toastBaggle: {result_baggle}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())



