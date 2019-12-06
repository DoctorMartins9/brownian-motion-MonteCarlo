import brownian_motion


# Main function
if __name__ == "__main__":
    example = brownian_motion.getRendimento()
    print('media: ' + str(example[0]) + '\n' 'varianza: ' + str(example[1]) )
    brownian_motion.moto_browniano_geometrico()