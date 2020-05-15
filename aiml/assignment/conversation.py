import aiml

# Create the kernel and learn AIML files
kernel = aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("load aiml b")

# Press CTRL-C to break this loop
while True:
    input_text = raw_input("Enter your message >> ")
    response = kernel.respond(input_text)
    print(response)


   