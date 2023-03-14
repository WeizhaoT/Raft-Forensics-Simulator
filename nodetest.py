from node import Block, node.Node


if __name__ == '__main__':
    node = node.Node(1, 5)
    chain1 = [Block(1, i, 0 if i == 1 else 1) for i in range(1, 11)]
    print(node.handle_append_entries(chain1))
    print(node.commit(2, 5))
    for b in node.blocks:
        print(b)
    print(node.commit(1, 5))
    print(node.commit(1, 5))
    for b in node.blocks:
        print(b)

    chain2 = [Block(1, i, 1) for i in range(3, 6)] + [Block(2, 6, 1)] + [Block(2, i, 2) for i in range(7, 10)]
    print('append 2: ', node.handle_append_entries(chain2))
    for b in node.blocks:
        print(b)

    chain2b = [Block(1, i, 1) for i in range(3, 6)] + [Block(2, 6, 1)] + [Block(2, i, 2) for i in range(7, 8)]
    print('append 2b: ', node.handle_append_entries(chain2b))
    for b in node.blocks:
        print(b)

    print(node.has(Block(1, 3, 1)))
    print(node.has(Block(1, 3, 1, '1')))
    print(node.has(Block(1, 5, 2)))
    print(node.has(Block(1, 5, 1)))

    # chain3 = [Block(3, 8, 2)] + [Block(3, i, 3) for i in range(9, 10)]
    # print('append 3: ', node.handle_append_entries(chain3))
    # for b in node.blocks:
    #     print(b)

    # print(node.commit(3, 8))
    # for b in node.blocks:
    #     print(b)

    # chain4 = [Block(3, i, 3) for i in range(12, 15)]
    # print('append 4: ', node.handle_append_entries(chain4))
    # for b in node.blocks:
    #     print(b)

    # print('Blocks since 2:')
    # for b in node.read_blocks_since(2):
    #     print(b)
