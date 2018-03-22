function y = l2lossForward( x,r )

delta = x - r;

y = sum(delta(:).^2);

y = y / (size(x,1)*size(x,2));

end

