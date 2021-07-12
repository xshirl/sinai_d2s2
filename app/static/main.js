function truncate(input) {
    if (input.length > 25) {
        input = input.substring(0, 26) + "..."
    }
    else {
        return input;
    }
}

 




