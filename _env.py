# -*- coding: utf-8 -*-

# 0 should not be chosen
NODE_INTERIOR = -1  # nodes located in the interior
# nodes located in the complement of (interior + frontier)
NODE_COMPLEMENTARY = -2
NODE_DIRICHLET = 1  # nodes with dirichlet boundary condition
NODE_NEUMANN = 2  # nodes with neumann boundary condition
NODE_ROBIN = 3  # nodes with robin boundary condition
