# File Objects

#############
## READING ##
#############

# first method for reading
f = open('text.txt','r')

print(f.name)
print(f.mode) # reading, writing, ...

f.close() # close the file to avoid leaks

# second method for reading

with open('text.txt','r') as f:
    
    size_to_read = 100
    f_contents = f.read()
    print(f_contents)
    # or
    f_contents = f.readlines()
    print(f_contents)
    # or 
    f_contents = f.readline()
    print(f_contents, end='')
    f_contents = f.readline()
    print(f_contents)

    # or (for larger files)

    for line in f:
        print(line,end='')

    # or 
    f_contents = f.read(size_to_read) # reads the first 100 characters

    while len(f_contents) > 0:
        print(f_contents)
        f_contents = f.read(size_to_read) 



#############
## WRITING ##
#############