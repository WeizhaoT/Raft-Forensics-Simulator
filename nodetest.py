import node


if __name__ == '__main__':
    v = node.Node(1, 5)
    chain1 = [node.Block(1, i, 0 if i == 1 else 1) for i in range(1, 11)]
    print(v.handle_append_entries(chain1))
    print(v.commit(2, 5))
    for b in v.blocks:
        print(b)
    print(v.commit(1, 5))
    print(v.commit(1, 5))
    for b in v.blocks:
        print(b)

    chain2 = [node.Block(1, i, 1) for i in range(3, 6)] + [node.Block(2, 6, 1)] + \
        [node.Block(2, i, 2) for i in range(7, 10)]
    print('append 2: ', v.handle_append_entries(chain2))
    for b in v.blocks:
        print(b)

    chain2b = [node.Block(1, i, 1) for i in range(3, 6)] + [node.Block(2, 6, 1)] + \
        [node.Block(2, i, 2) for i in range(7, 8)]
    print('append 2b: ', v.handle_append_entries(chain2b))
    for b in v.blocks:
        print(b)

    print(v.has(node.Block(1, 3, 1)))
    print(v.has(node.Block(1, 3, 1, '1')))
    print(v.has(node.Block(1, 5, 2)))
    print(v.has(node.Block(1, 5, 1)))

    # chain3 = [node.Block(3, 8, 2)] + [node.Block(3, i, 3) for i in range(9, 10)]
    # print('append 3: ', node.handle_append_entries(chain3))
    # for b in node.blocks:
    #     print(b)

    # print(node.commit(3, 8))
    # for b in node.blocks:
    #     print(b)

    # chain4 = [node.Block(3, i, 3) for i in range(12, 15)]
    # print('append 4: ', node.handle_append_entries(chain4))
    # for b in node.blocks:
    #     print(b)

    # print('node.Blocks since 2:')
    # for b in node.read_blocks_since(2):
    #     print(b)
